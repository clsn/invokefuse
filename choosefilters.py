#!/usr/bin/env python3

# Motivation: browse the "boards" created by InvokeAI as if they were subdirectories.
# To extend: do better than that.

# Development from invokefuse, per some discussions with SkunkWorxDark on Discord

# Plan for this time around: have different "filters" which you can apply
# in any order, any subset thereof, forming paths like
# /_BOARDS/landscapes/_MODELS/analogMadness70/_IMAGES

# by Mark Shoulson, 2024
# Still rough!

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

datere = re.compile('\\d{4}-\\d{2}-\\d{2}$')

METADATA = ".META"

# Name for empty values
NONE = "NONE"

# Filters.  To be applied as nested SQL queries.  Wait, nested?  Oh, that works
# only if the nested queries select ALL the columns and only the last query
# selects the individual item.  There are two queries for each filter, one
# for the filter and one for its value.

filters = {
    "_BOARDS": {"endquery": "SELECT DISTINCT full_board_name FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE IIF(:board IS NULL, "
                             "full_board_name IS NULL, "
                             "full_board_name = :board)"),
                "variable": "board",
                },
    "_MODELS": {"endquery": "SELECT DISTINCT model_name FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE IIF(:model IS NULL, "
                             "model_name IS NULL, "
                             "model_name = :model)"),
                "variable": "model",
                },
    "_DATES" : {"endquery": "SELECT DISTINCT DATE(created_at) FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE IIF(:date IS NULL, "
                             "created_at IS NULL, "
                             "DATE(created_at) = :date)"),
                # Maybe some way to specify a range?  Could do _AFTER and
                # _BEFORE.
                "variable": "date",
                },
    "_BEFORE" : {"endquery": "SELECT DISTINCT DATE(created_at) FROM ({})",
                 "midquery": ("SELECT * FROM ({}) WHERE "
                              "DATE(created_at) <= :before"),
                 "variable": "before",
                 "invisible": True,
                 },
    "_AFTER" : {"endquery": "SELECT DISTINCT DATE(created_at) FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE "
                             "DATE(created_at) >= :after"),
                "variable": "after",
                "invisible": True,
                },
    "_IMAGES": {"endquery": "SELECT image_name FROM ({})",
                "midquery": None,
                "variable": None
                },
}


ImageTbl_cmd = f"""select images.*, board_images.board_id, board_name, coalesce(board_name, '{UNSORTED}') as full_board_name, json_extract(metadata, '$.positive_prompt') as positive_prompt, coalesce(json_extract(metadata, '$.model.model_name'), models.name, json_extract(metadata, '$.model.key')) as model_name from images left join board_images on images.image_name=board_images.image_name left join boards on board_images.board_id=boards.board_id left join models on json_extract(metadata, '$.model.key')=models.id"""

ImageTbl = "all_images_boards"

class ChooseInvokeFS(fuse.Operations):
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

    def destroy(self, *args, **kwargs):
        pass

    def is_root(self, path=None, pathelts=None):
        if pathelts is None:
            pathelts = getParts(path)
        return path == os.sep or len(pathelts) == 0 or pathelts == [os.sep]

    def is_directory(self, path=None, pathelts=None):
        if not pathelts:
            pathelts=getParts(path)
        # print(f"is_dir: {pathelts=}")
        # It's a directory unless its parent is "_IMAGES".
        # This is likely subject to change!
        if self.is_root(path, pathelts):
            return True
        if len(pathelts) <= 2:  # Top-level dirs
            return True
        try:
            fil = filters[pathelts[-2]]
            return bool(fil.get('midquery',None))
        except KeyError:
            return True         # parent was not a filter level

    def getpromptnames(self):
        # Populate/refresh the self.promptdict library.
        self.promptdict.clear()
        self.cursor.execute("""select distinct
                                replace(
                                 json_extract(metadata,
                                             '$.positive_prompt'),
                                '/', ' ') from images;""")
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
        if self.is_directory(pathelts=pe):
            return st           # same as root, fine.

        st['st_mode'] = stat.S_IFREG | 0o444
        st['st_nlink'] = 1
        # Most of the rest of this should work mainly like the
        # old invokefuse, though haven't worked out special-cases
        # like PROMPT.TXT

        # Special case!  The prompt text file!
        # XXXX!!! Do the prompt text file?
        # if pe[-1] == PROMPT:
        #         promptname = info.get("promptname", None)
        #         # If there's no promptname, let it fail later.
        #         prompt = self.getprompt(promptname)
        #         st['st_mode'] = stat.S_IFREG | 0o444
        #         st['st_nlink'] = 1
        #         # Careful!  It's the length IN BYTES!
        #         st['st_size'] = len(prompt.encode('utf-8'))
        #         return st

        # Going to bother doing metadata files?
        # if pe[-1].endswith(METADATA):
        #     image = pe[-1][:-len(METADATA)]
        #     query = "SELECT metadata FROM images WHERE image_name=?;"
        #     self.cursor.execute(query, [image])
        #     # Yes, this winds up fetching the data when statting and when
        #     # reading.  Tough.
        #     ss = self.cursor.fetchone()
        #     if ss is None:
        #         raise fuse.FuseOSError(errno.ENOENT)
        #     st['st_mode'] = stat.S_IFREG | 0o444
        #     st['st_nlink'] = 1
        #     # ss[0] might still be None, though!
        #     if ss[0] is None:
        #         st['st_size'] = 0
        #     else:
        #         st['st_size'] = len(ss[0].encode('utf-8'))
        #     return st
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
        st['st_nlink']=1
        return st

    def imgfile(self, name):
        return os.sep.join([self.imagesdir, name])

    def readlink(self, filename):
        # print("RdLink: ({0!r} ({1!r})".format(filename, self.rootdir))
        pe = getParts(filename)
        name = pe[-1]
        return self.imgfile(name)

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

    def readdir(self, path, offset):
        pe = getParts(path=path)
        # print(f"{pe=}")
        if not self.is_directory(pathelts=pe):
            raise fuse.FuseOSError(errno.ENOTDIR)
        # Always yield '.' and '..'... but only after we're sure there's
        # something!  We can't raise an exception after there has been
        # something yielded, can we?  Do they always have to be first?
        # Problem is that ENOENT doesn't seem to be happening.  Raising
        # an ENOENT doesn't seem to show an error message with ls.
        # I guess raise ENOTDIR since we're reading dirs anyway?
        yield '.'
        yield '..'

        # build nested query, working from TOP DOWN, not bottom up!!.
        active = []
        for k, v in filters.items():
            if not v.get("invisible", False):
                active.append(k)
        vals = {}
        query = ImageTbl        # innermost query goes to ImageTbl.
        if self.is_root(pathelts=pe):
            # SPECIAL CASE: root dir.  Yield the filters.
            yield from iter(active)
            return
        # OK, ok, calm down.  Building from the TOP DOWN is not the same
        # process for each level.  Deciding which query of the filter to
        # use depends ONLY on the LAST element, so it must be handled
        # specially!  Higher elements should always come in pairs, and
        # we'll always use the second query!
        pec = pe.copy()
        # pe will always start with '' for the root dir.  So even numbers
        # means a key-level, odd numbers a value-level, though I will not
        # check that way...
        pec.pop(0)              # pop off the root dir.
        while len(pec) > 2:
            # Things come in pairs until I'm down to the last one or
            # two elements.  Should be straightforward.
            curr = pec.pop(0)
            try:
                fil = filters[curr]
            except KeyError:
                raise fuse.FuseOSError(errno.ENOTDIR)
            try:
                active.remove(curr)
            except ValueError:
                pass
            # Other exceptions can bubble up for now?
            # Non-last, so we use the second query, and set the var.
            query = fil['midquery'].format(query)
            val = pec.pop(0)
            if val == NONE:
                val = None
            vals[fil['variable']] = val
        # And now we're at the bottom, only one or two levels to go.
        # Should still be a key level.
        curr = pec.pop(0)
        try:
            fil = filters[curr]
        except KeyError:
            raise fuse.FuseOSError(errno.ENOTDIR)
        try:
            active.remove(curr)
        except ValueError:
            pass
        if pec:                 # Does there remain a value?
            # If so, then what I return is the remaining filters!!!
            # UNLESS there is nothing there!
            # Add the midquery here.
            query = fil['midquery'].format(query)
            val = pec[0]
            if val == NONE:
                val = None
            vals[fil['variable']] = val
            # print(f"{query=}\n{vals=}")
            self.cursor.execute(f"SELECT COUNT(*) from ({query})", vals)
            row = self.cursor.fetchone()
            if row[0] <= 0:
                raise fuse.FuseOSError(errno.ENOTDIR)
            yield from iter(active)
            return
        else:
            query = fil['endquery'].format(query)
        # print(f"{query=}\n{vals=}")
        self.cursor.execute(query, vals)
        # found = False
        # Can I use something like "else" on the while for this?
        # Do we even want to raise this?  An empty dir is okay.
        while (row := self.cursor.fetchone()):
            # found = True
            v = row[0]
            if v is None:
                yield NONE
            else:
                yield v
        # if not found:
        #     raise fuse.FuseOSError(errno.ENOENT)
        return

    def read(self, path, size, offset, fh):
        # What's the FH?
        pe = getParts(path)
        if self.is_directory(pathelts=pe):
            raise fuse.FuseOSError(errno.EISDIR)
        # if info.get("promptname", None) and pe[-1] == PROMPT:
        #     prompt = self.getprompt(info['promptname'])
        #     bprompt = prompt.encode('utf8')
        #     return bprompt[offset:offset+size]
        elif pe[-1].endswith(METADATA):
            img = pe[-1][:-len(METADATA)]
            self.cursor.execute("SELECT metadata FROM images WHERE image_name = ?;",
                                [img])
            ss = self.cursor.fetchone()
            if ss is None:
                raise fuse.FuseOSError(errno.ENOENT)
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

    If rootdir is not specified, it is taken to be one level above the directory
    where the dbfile is.  If imagesdir is not specified, it is taken to be
    "rootdir/outputs/images".
    """)

if __name__ == '__main__':
    server = ChooseInvokeFS()
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
    fu = fuse.FUSE(server, mntpt,
                   foreground=getattr(server, 'foreground', False),
                   nothreads=True,
                   allow_other=getattr(server, 'allow_other', False),
                   allow_root=getattr(server, 'allow_root', False))

