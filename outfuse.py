#!/usr/bin/env python3

# Motivation: browse the "boards" created by InvokeAI as if they were subdirectories.
# To extend: do better than that.  I think here's what I'm seeing:

# - mnt/
#   - boards/   # XXXX NO, REMOVE THIS LEVEL.
#     - UNSORTED/
#       - models/
#         - <model1 name>/
#           - 1/  (might have to use some hash?)
#             - __PROMPT (file which contains the text of the prompt)
#             - <image1.png>
#             - <image2.png>
#             - <etc...>
#           - 2/
#             - (same as 1)
#           - (etc)/
#         - <model2 name>/
#           - (same substructure as model1)...
#       - prompts/
#         - 1/
#           - __PROMPT (file containing text of the prompt)
#           - <model1 name>/
#             - <image1.png>
#             - <image2.png>
#             - <etc...>
#           - <model2 name>/
#             - (same as model1)
#         - 2/
#           - (same substructure as 1)...
#     - <boardname1>/
#       - (same substructure as UNSORTED)...
#     - <boardname2>/
#       - (same substructure as UNSORTED)...
#     - (etc)/


import fuse
import os
import path
import sys
import stat
import errno
import tempfile
import sqlite3 as sqlite
from itertools import count

import base64
import hashlib
from functools import update_wrapper

def debugf(func):
    def f(self, *args, **kwargs):
        print(f"Entering {f.__name__}({args!r}, {kwargs!r})")
        rv = func(self, *args, **kwargs)
        print(f"Returning {rv!r}")
        return rv
    update_wrapper(f, func)
    return f

def getParts(path):
    """
    Return the slash-separated parts of a given path as a list
    """
    if path == os.sep:
        return [os.sep]
    else:
        return path.split(os.sep)

# I'm not likely to need THAT big a hash space.
HASHLEN = 10
PROMPTLEN = 20
import re
SanityRE = re.compile('[^A-Za-z0-9_]+')
def makehash(prompt):
    # Special-case out the NOPROMPT
    if prompt == NOPROMPT:
        return NOPROMPT
    shk = hashlib.shake_256(prompt.encode('utf-8'))
    ## Eep, can't use b64encode, it has / in it!
    # rv = base64.b64encode(shk.digest(HASHLEN)).decode('utf-8')
    # Hmm.  I understand why I'm using hashes and not the whole string.
    # But what about (sanitizing the string and) taking the first N chars
    # and then appending the has, so they're still unique and they give
    # you SOME idea of what each one was?  And I'm only using it for uniqueness,
    # I can replace the /...
    # rv = shk.hexdigest(HASHLEN)
    h = base64.b64encode(shk.digest(HASHLEN)).decode('utf-8')
    h = h.replace('/', '@')
    pr = SanityRE.sub(' ', prompt)[:PROMPTLEN]
    pr = pr.strip()
    rv = f"{pr}-{h}"
    return rv

# Name for the unsorted board
# It's YOUR responsibility to make sure these special names don't conflict
# with anything actually being used.
UNSORTED = "UNSORTED"

# Name for prompt file
PROMPT = "PROMPT.TXT"

# Name for no-model models
NOMODEL = "NO MODEL"

# Name for no-prompt prompts, if any
NOPROMPT = "NO PROMPT"

# Ugh, such a pain.  OK, for reference:

# select images.*, board_images.board_id, board_name from images left join board_images on images.image_name=board_images.image_name left join boards on board_images.board_id=boards.board_id;

# There's surely a right way in SQL to make a View or something like that,
# but I don't want to do anything that's even adjacent to altering the
# database file, so I'll just put that in a string and use it as a
# subquery, k?

# Mmm, ok, so I tested (tried creating a temp view in a read-only file) and
# it looks like a temporary view does NOT try to change the DB, so it
# should be safe.  Taking off the parens and making it a view, but leaving
# the f-strings, whatever...
# It's SOO much simpler just to make an "UNSORTED" board than to hassle
# with the IS NULL situation.

ImageTbl_cmd = f"select images.*, board_images.board_id, board_name, coalesce(board_name, '{UNSORTED}') as full_board_name from images left join board_images on images.image_name=board_images.image_name left join boards on board_images.board_id=boards.board_id"

ImageTbl = "all_images_boards"

class InvokeOutFS(fuse.Operations):
    def init(self, *args, **kwargs):
        self.dbfile = os.path.abspath(self.dbfile)
        try:
            self.connection = sqlite.connect(self.dbfile)
        except sqlite.OperationalError as e:
            print("Error: %s"%e)
            exit(50)
        self.cursor = self.connection.cursor()
        self.cursor.execute(f"CREATE TEMPORARY VIEW {ImageTbl} as {ImageTbl_cmd};")
        if not getattr(self, "rootdir", None):
            self.rootdir = os.sep.join(self.dbfile.split(os.sep)[:-2])
        self.promptdict = {}
        self.indent=0

    def destroy(self, *args, **kwargs):
        pass

    def is_root(self, path=None, pathelts=None):
        if pathelts is None:
            pathelts = getParts(path)
        return path == os.sep or len(pathelts) == 0 or pathelts == [os.sep]

    def parseelts(self, pathelts):
        # oh, whatever, it returns a dict:
        # I guess distinguish /board from /board/"models" and board/"prompts" by
        # length?  No, still have to look at the last elt.  So add those in
        # {"board": board, "model": model, "promptname": promptname,
        #  "is_dir": boolean, "tree" : ("prompt" or "model")}
        # Leaves out or None what it don't know
        board = None
        model = None
        promptname = None
        is_dir = True
        tree = None
        # I thought this would be a good use (finally) of the match statement.
        # I was wrong.  Forget it.
        # Started with repeated ifs, I think I can nest...
        # Not sure why I should ever not see '' at the front of the list,
        # but it's happening?  Is it causing the problem?
        #if pathelts != [os.sep] and pathelts[0] != '':
        #    pathelts.insert(0, '')
        numelts = len(pathelts)
        if numelts >= 2:    # ['', board]
            board = pathelts[1]
            # Otherwise, not much beyond is_dir=True
            if numelts >= 3:    # ['', board, ("models"|"prompts")]
                tree = pathelts[2]
                if numelts >= 4: # ['', board, tree, (model|promptname)]
                    if tree == "models":
                        model = pathelts[3]
                    else:
                        promptname = pathelts[3]
                    if numelts >= 5: # ['', board, tree, 1stlev, 2ndlev]
                        if tree == "prompts":
                            # Special case!!
                            if pathelts[4] == PROMPT:
                                is_dir = False # !
                            else:
                                model = pathelts[4]
                        else:
                            promptname = pathelts[4]
                        if numelts > 5:
                            is_dir = False # At the image level now.
        return dict(board=board, model=model, promptname=promptname,
                    tree=tree, is_dir=is_dir)

    def is_directory(self, path=None, pathelts=None):
        if not pathelts:
            pathelts=getParts(path)
        info = self.parseelts(pathelts)
        return info['is_dir']

    def getpromptnames(self):
        # Populate/refresh the self.promptdict library.
        self.promptdict.clear()
        self.cursor.execute("select distinct "
                            "json_extract(metadata, '$.positive_prompt') from images;")
        while (batch := self.cursor.fetchmany()):
            for item in batch:
                p = item[0]
                if p:           # It's sometimes None?
                    self.promptdict[makehash(p)] = p

    def getprompt(self, promptname):
        try:
            return self.promptdict[promptname]
        except KeyError:
            # Refresh the promptdict and try again.
            self.getpromptnames()
            try:
                return self.promptdict[promptname]
            except KeyError:
                # OK to raise here?
                raise fuse.FuseOSError(fuse.ENOENT) # ?

    def getattr(self, path, fh=None):
        pe=getParts(path)
        st = dict(st_mode = stat.S_IFDIR | 0o555,
                  st_ino = 0,
                  st_dev = 0,
                  st_nlink = 2,
                  st_uid = 0,
                  st_gid = 0,
                  st_size = 4096,
                  st_atime = 0,
                  st_mtime = 0,
                  st_ctime = 0)
        if self.is_root(pathelts=pe):
            return st
        info = self.parseelts(pe)
        if info.get("is_dir", False):
            # XXXX Not confirming the existence of model or promptname subdir!
            return st           # same as root, fine.
        else:
            # Special case!  The prompt text file!
            if pe[-1] == PROMPT:
                promptname = info.get("promptname", None)
                # If there's no promptname, let it fail later.
                prompt = self.getprompt(promptname)
                st['st_mode'] = stat.S_IFREG | 0o444
                st['st_nlink'] = 1
                # Careful!  It's the length IN BYTES!
                st['st_size'] = len(prompt.encode('utf-8'))
                return st
            # Otherwise, this is presumably an image file, so a soft link.
            st['st_mode']=stat.S_IFLNK | 0o777
            st['st_nlink']=1
            imgname=pe[-1]
            st['st_size'] = len(imgname)
            # print(" IZImg ({0})".format(imgname))
            query="SELECT COUNT(*) FROM images WHERE image_name=?;"
            try:
                self.cursor.execute(query, [imgname])
                cnt=self.cursor.fetchone()
            except Exception as e:
                # self.DBG("Whoa, except getattr2: {0}".format(e))
                cnt=[0]
            if cnt[0]<1:
                # self.DBG("File not found.")
                raise fuse.FuseOSError(fuse.ENOENT)
        return st

    def readlink(self, filename):
        # print("RdLink: ({0!r} ({1!r})".format(filename, self.rootdir))
        pe = getParts(filename)
        name = pe[-1]
        return os.sep.join([self.rootdir, "outputs", "images", name])

    # Also, maybe for "all" boards (i.e. a combined board)  Possibly a bad idea,
    # unless optional at mount time.
    def listmodels(self, board, prompt=None):
        # Don't read from model_config; models might have been
        # deleted.  Also, there ARE images with NO MODEL!!  This
        # is okay!
        # OK, be careful.  When prompt is None, don't restrict by prompt
        # at all.  When prompt is NOPROMPT, restrict to having no prompt.
        # Note that this should be REAL PROMPTS and not the hashed promptname!
        if prompt is None:
            self.cursor.execute("SELECT DISTINCT "
                                "json_extract(metadata, '$.model.model_name') "
                                f"FROM {ImageTbl} "
                                "where full_board_name=?;", [board])
        elif prompt == NOPROMPT:
            self.cursor.execute("SELECT DISTINCT "
                                "json_extract(metadata, '$.model.model_name') "
                                f"FROM {ImageTbl} "
                                "WHERE json_extract(metadata, '$.positive_prompt') IS NULL AND "
                                "full_board_name=?;", [board])
        else:
            self.cursor.execute("SELECT DISTINCT "
                                "json_extract(metadata, '$.model.model_name') "
                                f"FROM {ImageTbl} "
                                "WHERE json_extract(metadata, '$.positive_prompt')=? "
                                "AND full_board_name=?;",
                                [prompt, board])
        # Maybe I should always yield NOMODEL.
        yield NOMODEL
        while (batch := self.cursor.fetchmany()):
            for r in batch:
                if not r or not r[0]:
                    pass        # ???? XXXX
                else:
                    yield r[0]

    def listprompts(self, board, model=None, *, hash=False):
        # As above, for prompts instead of models.  Let's say this yields
        # REAL PROMPTS and the caller has to hash to promptnames as needed.
        # But for some reason "yield from" works and a for loop that hashes
        # and then yields doesn't.  So I guess hash here, optionally?
        if model is None:
            self.cursor.execute("SELECT DISTINCT "
                                "json_extract(metadata, '$.positive_prompt') "
                                f"FROM {ImageTbl} "
                                "WHERE full_board_name=?;", [board])
        elif model == NOMODEL:
            self.cursor.execute("SELECT DISTINCT "
                                "json_extract(metadata, '$.positive_prompt') "
                                f"FROM {ImageTbl} "
                                "WHERE json_extract(metadata, '$.model.model_name') is NULL "
                                "AND full_board_name=?;", [board])
        else:
            self.cursor.execute("SELECT DISTINCT "
                                "json_extract(metadata, '$.positive_prompt') "
                                f"FROM {ImageTbl} "
                                "WHERE json_extract(metadata, '$.model.model_name')=? "
                                "AND full_board_name=?;",
                                [model, board])
        # Maybe I should yield NOPROMPT no matter what
        yield NOPROMPT
        while (batch := self.cursor.fetchmany()):
            for r in batch:
                if not r or not r[0]:
                    pass        #  ???? XXXX
                else:
                    if hash:
                        yield makehash(r[0])
                    else:
                        yield r[0]

    def listimages(self, board, prompt=None, model=None):
        # Use REAL PROMPT, and None vs NOMODEL and NOPROMPT as the others.
        # Maybe can be a LITTLE more efficient.
        # There's also a right way do to THIS in sqlite, isn't there?
        # And I'm not doing it?
        rprompt = rmodel = ""
        params=[]
        if prompt is not None:
            if prompt == NOPROMPT:
                rprompt = "json_extract(metadata, '$.positive_prompt') IS NULL"
            else:
                rprompt = "json_extract(metadata, '$.positive_prompt') = ?"
                params.append(prompt)
        if model is not None:
            if model == NOMODEL:
                rmodel = "json_extract(metadata, '$.model.model_name') IS NULL"
            else:
                rmodel = "json_extract(metadata, '$.model.model_name') = ?"
                params.append(model)
        if rprompt and rmodel:
            restrict = f"{rprompt} AND {rmodel}"
        else:
            restrict = rprompt or rmodel
        if restrict:
            restrict = " AND " + restrict
        self.cursor.execute(f"SELECT image_name from {ImageTbl} WHERE "
                            f"full_board_name = ?{restrict};",
                            [board] + params)
        while (batch := self.cursor.fetchmany()):
               for r in batch:
                   if not r or not r[0]:
                       # XXXX RAISE ERROR?
                       yield ""
                   else:
                       yield r[0]

    def readdir(self, path, offset):
        pe = getParts(path=path)
        info = self.parseelts(pe)
        if not info.get('is_dir', False):
            raise fuse.FuseOSError(fuse.ENOTDIR)
        yield '.'
        yield '..'
        if self.is_root(path=path):
            # Always yield the unsorted dir
            yield UNSORTED
            self.cursor.execute("SELECT DISTINCT board_name FROM boards;")
            l = self.cursor.fetchall()
            for r in l:
                yield r[0]
            return
        # we SHOULD have the board at this point.
        board = info.get('board', None)
        if board is None:
            # Problem, right?
            raise fuse.FuseOSError(ENOENT) # ??
        if not info.get('tree', None):
            # the board is supplied first, then the tree, so we must know
            # the board and that's the level we're on.
            # Confirm that it exists this time?
            self.cursor.execute(f"SELECT count(*) FROM {ImageTbl} "
                                "WHERE full_board_name=?;",
                                [board])
            res = self.cursor.fetchone()
            if res[0] <= 0:
                raise fuse.FuseOSError(fuse.ENOENT) # ?
            # We're at the tree level, so the only things to
            # return are the two possible trees:
            yield "models"
            yield "prompts"
        elif info['tree'] == "models":
            # We're in one of two branches now: prompts or models.  If we
            # know one but not the other, list the other.  If we know
            # neither, list the one we don't know.
            # My structure isn't well-suited for good code-reuse.  Oh well.
            if (model := info.get('model', None)):
                # Are we at the lowest level, knowing both model and
                # prompt?
                if (promptname := info.get('promptname', None)):
                    # We're in the model tree at the bottom, need to
                    # output the PROMPT file too.
                    prompt = self.getprompt(promptname) # XXX exception here?
                    yield PROMPT
                    yield from self.listimages(board, model=model, prompt=prompt)
                else:
                    # we know the model but not the prompts.  I think we
                    # *should* restrict to prompts that are actually found
                    # in that model.
                    for p in self.listprompts(board, model):
                        yield makehash(p)
            else:
                # We are in models tree, but don't know the model;
                # have to list those.
                yield from self.listmodels(board)
        elif info['tree'] == 'prompts':
            if (prompt := info.get('promptname', None)):
                if prompt != NOPROMPT:
                    prompt = self.getprompt(prompt) # raises error here?  probably wrong?
                # Are we at the bottom now?
                if (model := info.get('model', None)):
                    yield from self.listimages(board, model=model, prompt=prompt)
                else:
                    # Have to list the models for this prompt.
                    # Also the PROMPT entry!
                    yield PROMPT
                    yield from self.listmodels(board, prompt)
            else:
                # Need to list the prompts, but hashed!
                for p in self.listprompts(board):
                    yield makehash(p)

    # I actually have to have a read() for the prompt.

    def read(self, path, size, offset, fh):
        # What's the FH?
        pe = getParts(path)
        info = self.parseelts(pe)
        if info.get("is_dir", False):
            raise fuse.FuseOSError(fuse.EISDIR)
        if not info.get("promptname", None) or pe[-1] != PROMPT:
            raise fuse.FuseOSError(fuse.EBADF) # ?
        prompt = self.getprompt(info['promptname'])
        bprompt = prompt.encode('utf8')
        return bprompt[offset:offset+size]

    mknod = unlink = write = mkdir = release = open = truncate = utime = None

    symlink = None
    link = None

    rmdir = chmod = None

def usage():
    print(f"""
    -o dbfile=$PWD/databases/invokeai.db
    """)

if __name__ == '__main__':
    server = InvokeOutFS()
    server.path = os.getcwd()
    # Simple parsing.  Maybe I should do better?
    if sys.argv[1] == "--help":
        usage()
        sys.exit(0)
    if sys.argv[1].startswith("-o"):
        opts = sys.argv.pop(1)
        if opts == '-o':
            opts = sys.argv.pop(1)
        else:
            # ?? wtf??
            opts = opts[2:]
        for opt in opts.split(","):
            try:
                nam, val = opt.split('=', 2)
            except ValueError:
                nam, val = opt, True
            if not val:
                val = True
            setattr(server, nam, val)
    mntpt = os.path.abspath(sys.argv[1])
    fu = fuse.FUSE(server, mntpt, foreground=hasattr(server,'foreground'),
                   nothreads=True)

