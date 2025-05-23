#!/usr/bin/env python3

# Motivation: browse the "boards" created by InvokeAI as if they were subdirectories.
# To extend: do better than that.  Basically, this is the idea:

# - root/
#   - UNSORTED
#     - models
#       - <model name1>
#         - <"hashed" prompt1>
#           - <image1.png>
#           - <image2.png>
#           - ...
#           - PROMPT.txt
#         - <"hashed" prompt2>
#           - (as above)
#         - ...
#         - NO PROMPT
#           - (as above)
#       - <model name2>
#         - (as above)
#       - <model name3>
#         - (as above)
#       - ...
#       - NO MODEL
#         - (as above)
#   - prompts
#     - <"hashed" prompt1>
#       - PROMPT.TXT
#       - <model name1>
#         - <image1.png>
#         - <image2.png>
#         - ...
#       - <model name2>
#         - (as above)
#       - ...
#       - NO MODEL
#         - (as above)
#     - <"hashed" prompt2>
#       - (as above)
#     - ...
#     - NO PROMPT
#       - (as above)
# - <board name1>
#   - (as above)
# - <board name2>
#   - (as above)
#   - ...

# by Mark Shoulson, 2023-2024
# Still rough!

# Added in dates, though badly.  Only really makes sense if you "skip" the
# existing levels with "ALL", if you want to skip them.

# How about better prompt searching?

import fuse
import os
import path
import sys
import stat
import errno
import tempfile
import re
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
HASHLEN = 9                     # multiple of 3 means no =-padding.
PROMPTLEN = 20
import re
SanityRE = re.compile('[^A-Za-z0-9_]+')
def makehash(prompt):
    # Special-cases
    if prompt == NOPROMPT:
        return NOPROMPT
    if prompt == ALL:
        return ALL
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

# These should be defined constants

MODELS = "models"
PROMPTS = "prompts"
DATES = "dates"
# User's responsibility not to use this form for board-names!
datere = re.compile('\\d{4}-\\d{2}-\\d{2}$')

METADATA = ".META"

# Name for no-model models
NOMODEL = "NO MODEL"

# Name for no-prompt prompts, if any
NOPROMPT = "NO PROMPT"

# Special token for "ALL"; may be removed from user's sight.
ALL = "ALL"                     # "*" maybe better?

LIKE = "LIKE"

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

ImageTbl_cmd = f"""select images.*, board_images.board_id, board_name, coalesce(board_name, '{UNSORTED}') as full_board_name, json_extract(metadata, '$.positive_prompt') as positive_prompt, coalesce(json_extract(metadata, '$.model.model_name'), models.name, json_extract(metadata, '$.model.key')) as model_name from images left join board_images on images.image_name=board_images.image_name left join boards on board_images.board_id=boards.board_id left join models on json_extract(metadata, '$.model.key')=models.id where is_intermediate is false"""

ImageTbl = "all_images_boards"

class InvokeOutFS(fuse.Operations):
    def init(self, *args, **kwargs):
        self.dbfile = os.path.abspath(self.dbfile)
        try:
            self.connection = sqlite.connect(self.dbfile)
            self.cursor = self.connection.cursor()
        except sqlite.OperationalError as e:
            print("Error: %s"%e)
            exit(50)
        # Invoke 3.x does not have a models table.
        # CREATE TEMP TABLE IF NOT EXISTS doesn't work.
        self.cursor.execute("""SELECT count(*) FROM sqlite_master
                            WHERE type='table' and name='models';""")
        r = self.cursor.fetchone()
        if not r or r[0] < 1:
            # Temp tables don't write to the DB file either.
            self.cursor.execute("""CREATE TEMPORARY TABLE
                            models (key text, id text, name text);""")
        self.cursor.execute(f"CREATE TEMPORARY VIEW {ImageTbl} as {ImageTbl_cmd};")
        if not getattr(self, "rootdir", None):
            self.rootdir = os.sep.join(self.dbfile.split(os.sep)[:-2])
        if not getattr(self, 'imagesdir', None):
            self.imagesdir = os.sep.join([self.rootdir, "outputs", "images"])
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
        # print(f"parseelts({pathelts=}) -> ", end="")
        board = None
        model = None
        promptname = None
        is_dir = True
        tree = None
        list_dates = None
        day = None
        like = None
        # I thought this would be a good use (finally) of the match statement.
        # I was wrong.  Forget it.
        # Started with repeated ifs, I think I can nest...
        # Not sure why I should ever not see '' at the front of the list,
        # but it's happening?  Is it causing the problem?
        #if pathelts != [os.sep] and pathelts[0] != '':
        #    pathelts.insert(0, '')
        # Special casing for dates?  Chop them off the end, too!
        if pathelts[-1] == DATES:
            # it's a dir, asking for the dates at whatever level.
            is_dir = True
            list_dates=True
            pathelts = pathelts[:-1]
        elif datere.match(pathelts[-1]):
            # it's a dir, asking for the images from that date.  Yes,
            # only the images, don't get more complicated than that.
            # It is your responsibility not to name boards with names that
            # look like dates!  Or else change the RE!
            is_dir = True
            day = pathelts[-1]
            pathelts = pathelts[:-1]
        elif len(pathelts) > 1 and pathelts[-2] == LIKE:
            # I think I'll require "like" to be on the next-to-bottom level.
            # You can never list a dir with LIKE, but it lists prompts
            # underneath it.
            like = pathelts[-1]
            is_dir = True
            # Chop them off the end.  No, only the LIKE part.
            # On your head be it if you put "like" in the wrong place.
            pathelts = pathelts[:-2]
        # You know what??  No.  You want to use them without a board or a model,
        # you use ALL.  That's what it's for.
        # Chop 'em out ANYWHERE the magic term appears!!!
        pathelts = [_ for _ in pathelts if (_ != DATES and
                                            not datere.match(_))]
        try:
            ind = pathelts.index(LIKE)
            pathelts.pop(ind)
            pathelts.pop(ind)   # take out the next element too.
        except (IndexError,ValueError):
            pass
        numelts = len(pathelts)
        if numelts >= 2:    # ['', board]
            board = pathelts[1]
            # Otherwise, not much beyond is_dir=True
            if numelts >= 3:    # ['', board, ("models"|"prompts"|"dates")]
                tree = pathelts[2]
                if numelts >= 4: # ['', board, tree, (model|promptname)]
                    if tree == MODELS:
                        model = pathelts[3]
                    # elif tree == "dates": # dates maybe are very different?
                    #     pass
                    else:  #  tree == PROMPTS:
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
                    tree=tree, is_dir=is_dir, list_dates=list_dates,
                    day=day, like=like)

    def is_directory(self, path=None, pathelts=None):
        if not pathelts:
            pathelts=getParts(path)
        info = self.parseelts(pathelts)
        return info['is_dir']

    def getpromptnames(self):
        # Populate/refresh the self.promptdict library.
        self.promptdict.clear()
        self.cursor.execute("""select distinct
                                replace(
                                 json_extract(metadata,
                                             '$.positive_prompt'),
                                '/', ' ') from images where is_intermediate is false;""")
        while (batch := self.cursor.fetchmany()):
            for item in batch:
                p = item[0]
                if p:           # It's sometimes None?
                    self.promptdict[makehash(p)] = p

    def getprompt(self, promptname):
        if promptname == ALL:
            return ALL
        if promptname is None:
            return None
        try:
            return self.promptdict[promptname]
        except KeyError:
            # Refresh the promptdict and try again.
            self.getpromptnames()
            try:
                return self.promptdict[promptname]
            except KeyError:
                # OK to raise here?
                # raise fuse.FuseOSError(errno.ENOENT) # ?
                # No, maybe it's the full prompt!
                return promptname

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
            # UNLESS it's a metadata file!!!
            if pe[-1].endswith(METADATA):
                image = pe[-1][:-len(METADATA)]
                query = "SELECT metadata FROM images WHERE image_name=?;"
                self.cursor.execute(query, [image])
                # Yes, this winds up fetching the data when statting and when
                # reading.  Tough.
                ss = self.cursor.fetchone()
                if ss is None:
                    raise fuse.FuseOSError(errno.ENOENT)
                st['st_mode'] = stat.S_IFREG | 0o444
                st['st_nlink'] = 1
                # ss[0] might still be None, though!
                if ss[0] is None:
                    st['st_size'] = 0
                else:
                    st['st_size'] = len(ss[0].encode('utf-8'))
                return st
            # Set the date stuff on the link?  The user can always just
            # add `-L` to the ls command...
            imgname=pe[-1]
            query="SELECT COUNT(*) FROM images WHERE image_name=?;"
            try:
                self.cursor.execute(query, [imgname])
                cnt=self.cursor.fetchone()
            except Exception as e:
                # self.DBG("Whoa, except getattr2: {0}".format(e))
                cnt=[0]
            if cnt[0]<1:
                # self.DBG("File not found.")
                raise fuse.FuseOSError(errno.ENOENT)
            # It's a link UNLESS we are working in real-file mode!
            if getattr(self, 'real_files', False):
                st['st_mode'] = stat.S_IFREG | 0o444
                st['st_size'] = len(imgname)
                # I guess go get the data from the real thing.  The metadata
                # in the db doesn't have the size anyway.
                fst = os.stat(self.imgfile(imgname))
                # Let exceptions bubble up, I guess.
                # reuse the inode number and stuff.  Oww, I have to do
                # it by hand?  I refuse.
                for k in ['st_ino', 'st_dev', 'st_uid', 'st_gid', 'st_size',
                          'st_atime', 'st_mtime', 'st_ctime']:
                    st[k] = fst[getattr(stat, k.upper())]
            else:
                st['st_mode']=stat.S_IFLNK | 0o777
                # I can still set times and stuff right?
                # I suppose I could just use ls -L for this,
                # but still.
                query = "SELECT unixepoch(created_at) FROM images WHERE image_name=?;"
                self.cursor.execute(query, [imgname])
                cdate, = self.cursor.fetchone()
                for k in ['st_atime', 'st_mtime', 'st_ctime']:
                    st[k] = cdate
            st['st_nlink']=1
        return st

    def imgfile(self, name):
        return os.sep.join([self.imagesdir, name])

    def readlink(self, filename):
        # print("RdLink: ({0!r} ({1!r})".format(filename, self.rootdir))
        pe = getParts(filename)
        name = pe[-1]
        return self.imgfile(name)

    def listmodels(self, board, *, prompt=None):
        # Don't read from model_config; models might have been
        # deleted.  Also, there ARE images with NO MODEL!!  This
        # is okay!
        # OK, be careful.  When prompt is None, don't restrict by prompt
        # at all.  When prompt is NOPROMPT, restrict to having no prompt.
        # Note that this should be REAL PROMPTS and not the hashed promptname!
        # print(f"listmodels({board=}, {prompt=})")
        # It's safe to use these f-strings, I'm only including my own
        # constants.
        query = f"""SELECT DISTINCT model_name
        FROM {ImageTbl}
        WHERE
        (full_board_name = :boardname OR :boardname = '{ALL}')
        AND IIF(:prompt IS NULL OR :prompt = '{ALL}', TRUE,
                IIF(:prompt = '{NOPROMPT}',
                    positive_prompt IS NULL,
                    replace(
                        positive_prompt,
                        '/', ' ') = :prompt))
        """
        if prompt:
            prompt = prompt.replace('/', ' ')
        self.cursor.execute(query, {"boardname":board, "prompt":prompt})
        # Maybe I should always yield NOMODEL.
        yield NOMODEL
        while (batch := self.cursor.fetchmany()):
            for r in batch:
                if not r or not r[0]:
                    pass        # ???? XXXX
                else:
                    yield r[0]

    def listprompts(self, board, *, model=None, hash=False, like=None):
        # As above, for prompts instead of models.  Let's say this yields
        # REAL PROMPTS and the caller has to hash to promptnames as needed.
        # But for some reason "yield from" works and a for loop that hashes
        # and then yields doesn't.  So I guess hash here, optionally?
        # print(f"listprompts({board=}, {model=}, {hash=}, {like=})")
        query = f"""SELECT DISTINCT
            replace(positive_prompt, '/', ' ') as prm
        FROM {ImageTbl}
        WHERE
        (full_board_name = :boardname OR :boardname = '{ALL}')
        AND IIF(:model IS NULL OR :model = '{ALL}', TRUE,
                IIF(:model = '{NOMODEL}',
                    model_name IS NULL,
                    model_name = :model))
        AND (:like IS NULL OR prm LIKE :like)
        """
        self.cursor.execute(query, {"boardname":board,
                                    "model":model,
                                    "like":like})
        # Maybe I should yield NOPROMPT no matter what
        # Not if there's a "like", certainly!  Maybe not in other cases.
        if not like:
            yield NOPROMPT
        while (batch := self.cursor.fetchmany()):
            for r in batch:
                if not r or not r[0]:
                    pass        #  ???? XXXX
                else:
                    if hash and not like: # like negates hash
                        yield makehash(r[0])
                    else:
                        yield r[0]

    def listdates(self, **info):
        # print(f"listdates({info=})")
        # Just feed it in the info, okay?  Dates are different.
        # I'm just going to ASSUME list_dates is True, why else are you
        # calling this?
        # print(f"listdates(**{info=})")
        # This actually isn't QUITE accurate, since it doesn't take timezone
        # vs UTC into account.  But that's close enough.
        query = f"""SELECT DISTINCT DATE(created_at) FROM {ImageTbl} WHERE
        (full_board_name = :board OR :board = '{ALL}' OR :board IS NULL) AND
        IIF(:model IS NULL OR :model = '{ALL}', TRUE,
            IIF(:model = '{NOMODEL}',
                model_name IS NULL,
                model_name = :model)) AND
        IIF(:prompt IS NULL OR :prompt = '{ALL}', TRUE,
            IIF(:prompt = '{NOPROMPT}',
                positive_prompt IS NULL,
                replace(positive_prompt, '/', ' ') = :prompt))
        """
        self.cursor.execute(query, {"board": info.get('board', None),
                                    "model": info.get('model', None),
                                    "prompt":
                                    self.getprompt(info.get('promptname', None))})
        while (batch := self.cursor.fetchmany()):
               for r in batch:
                   if not r or not r[0]:
                       # XXXX RAISE ERROR?
                       yield ""
                   else:
                       yield r[0]

    def listimages(self, board, *, prompt=None, model=None, day=None):
        # Use REAL PROMPT, and None vs NOMODEL and NOPROMPT as the others.
        # Maybe can be a LITTLE more efficient.
        # print(f"listimages({board=}, {prompt=}, {model=}, {day=})")
        rprompt = rmodel = ""
        restrict = "full_board_name=?"
        query = f"""SELECT image_name FROM {ImageTbl} WHERE
        (full_board_name = :board OR :board = '{ALL}' OR :board IS NULL) AND
        IIF(:model IS NULL OR :model = '{ALL}', TRUE,
            IIF(:model = '{NOMODEL}',
                model_name IS NULL,
                model_name = :model)) AND
        IIF(:prompt IS NULL OR :prompt = '{ALL}', TRUE,
            IIF(:prompt = '{NOPROMPT}',
                positive_prompt IS NULL,
                replace(positive_prompt,'/',' ') = :prompt)) AND
        (DATE(created_at) = :day OR :day IS NULL)
        """
        if prompt:
            prompt = prompt.replace('/', ' ')
        self.cursor.execute(query, {"board":board,
                                    "model":model,
                                    "prompt":prompt,
                                    "day":day})
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
        # print(info)
        day = info['day']
        if not info.get('is_dir', False):
            raise fuse.FuseOSError(errno.ENOTDIR)
        yield '.'
        yield '..'
        # "dates" takes precedence over everything, and ONLY yields dates,
        # with images directly under that!!  No other tree stuff!
        if info['list_dates']:
            yield from self.listdates(**info)
            return
        # Dates always list images, never anything else.
        # XXXXX AUGH FAIL!  I rely on size of path to determine that these
        # XXXXX aren't directories!
        # You know what?  Fine.  Users should be using ALL to skip levels,
        # not sticking "dates" in the wrong place.
        if info['day']:
            if info['promptname']:
                prompt = self.getprompt(info['promptname'])
            else:
                prompt = None
            yield from self.listimages(board=info['board'], model=info['model'],
                                       prompt=prompt, day=info['day'])
            return              # And don't do any more!
        if self.is_root(path=path):
            # Always yield the unsorted dir
            yield UNSORTED
            if getattr(self, "incl_all", False):
                yield ALL
            self.cursor.execute("SELECT DISTINCT board_name FROM boards;")
            l = self.cursor.fetchall()
            for r in l:
                # What if a board name has a '/' in it???  User's problem.
                yield r[0]
            return
        # we SHOULD have the board at this point.
        board = info.get('board', None)
        restrict = "full_board_name=?"
        if board == ALL:
            restrict = "TRUE OR " + restrict
        if board is None:
            # Problem, right?
            raise fuse.FuseOSError(ENOENT) # ??
        if not info.get('tree', None):
            # the board is supplied first, then the tree, so we must know
            # the board and that's the level we're on.
            # Confirm that it exists this time?
            self.cursor.execute(f"SELECT count(*) FROM {ImageTbl} "
                                f"WHERE {restrict};",
                                [board])
            res = self.cursor.fetchone()
            if res[0] <= 0:
                raise fuse.FuseOSError(errno.ENOENT) # ?
            # We're at the tree level, so the only things to
            # return are the two possible trees:
            yield MODELS
            yield PROMPTS
        elif info['tree'] == MODELS:
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
                    if not (hasattr(self, 'no_prompt') and getattr(self, 'no_prompt')):
                        yield PROMPT
                    yield from self.listimages(board, model=model, prompt=prompt, day=day)
                else:
                    # we know the model but not the prompts.  I think we
                    # *should* restrict to prompts that are actually found
                    # in that model.
                    if getattr(self, 'incl_all', False):
                        yield ALL
                    yield from self.listprompts(board, model=model, hash=True,
                                                like=info['like'])
            else:
                # We are in models tree, but don't know the model;
                # have to list those.
                if getattr(self, 'incl_all', False):
                    yield ALL
                yield from self.listmodels(board)
        elif info['tree'] == PROMPTS:
            if (prompt := info.get('promptname', None)):
                if prompt != NOPROMPT:
                    prompt = self.getprompt(prompt) # raises error here?  probably wrong?
                # Are we at the bottom now?
                if (model := info.get('model', None)):
                    yield from self.listimages(board, model=model,
                                               prompt=prompt, day=day)
                else:
                    # Have to list the models for this prompt.
                    # Also the PROMPT entry!
                    if not (hasattr(self, 'no_prompt') and getattr(self, 'no_prompt')):
                        yield PROMPT
                    if getattr(self, 'incl_all', False):
                        yield ALL
                    yield from self.listmodels(board, prompt=prompt)
            else:
                if getattr(self, 'incl_all', False):
                    yield ALL
                yield from self.listprompts(board, hash=True, like=info['like'])

    # I actually have to have a read() for the prompt and metadata

    def read(self, path, size, offset, fh):
        # What's the FH?
        pe = getParts(path)
        info = self.parseelts(pe)
        if info.get("is_dir", False):
            raise fuse.FuseOSError(errno.EISDIR)
        if info.get("promptname", None) and pe[-1] == PROMPT:
            prompt = self.getprompt(info['promptname'])
            bprompt = prompt.encode('utf8')
            return bprompt[offset:offset+size]
        elif pe[-1].endswith(METADATA):
            img = pe[-1][:-len(METADATA)]
            self.cursor.execute("SELECT metadata FROM images WHERE image_name = ?;",
                                [img])
            ss = self.cursor.fetchone()
            if ss is None:
                raise fuse.FuseOSError(errno.EBADF)
            # Could still have ss[0] being None.
            if ss[0] is None:
                val = b""
            else:
                val = ss[0].encode('utf8')
            return val[offset:offset+size]
        else:
            # In real_files mode, we do it for realz.
            if getattr(self, 'real_files', False):
                img = pe[-1]
                imgfile = self.imgfile(img)
                # Let exceptions bubble up?
                with open(imgfile, "rb") as lfh:
                    lfh.seek(offset)
                    buf = lfh.read(size)
                    return buf
            raise fuse.FuseOSError(errno.EBADF) # ?

    mknod = unlink = write = mkdir = release = open = truncate = utime = None

    symlink = None
    link = None

    rmdir = chmod = None

def usage():
    print(f"""
    -o dbfile=$PWD/databases/invokeai.db ~/mnt

    options include 'foreground', 'allow_other',
    'imagesdir=ABSOLUTEPATH'   (where your images are)
    'rootdir=ABSOLUTEPATH'    (root dir for invoke)
    'real_files'  (emulate "real" files instead of symbolic links.)
    'incl_all'   (list ALL explicitly in directories)
    'no_prompt'   (omit {PROMPT} files from directories)

    If rootdir is not specified, it is taken to be one level above the directory
    where the dbfile is.  If imagesdir is not specified, it is taken to be
    "rootdir/outputs/images".
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
    fu = fuse.FUSE(server, mntpt, foreground=getattr(server, 'foreground', False),
                   nothreads=True, allow_other=getattr(server, 'allow_other', False),
                   allow_root=getattr(server, 'allow_root', False))

