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
        # print(f"Entering {f.__name__}({args!r}, {kwargs!r})")
        rv = func(self, *args, **kwargs)
        #print(f"Returning from {f.__name__}: {rv!r}")
        print(f"{f.__name__}({args!r}, {kwargs!r}) ->\n   {rv!r}")
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
NOPROMPT = 'NO PROMPT'
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

THISQUERY = "THISQUERY"

# Filters.  To be applied as nested SQL queries.  Wait, nested?  Oh, that works
# only if the nested queries select ALL the columns and only the last query
# selects the individual item.  There are two queries for each filter, one
# for the filter and one for its value.

FILTERS = {
    "_BOARDS": {"endquery": "SELECT DISTINCT full_board_name FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE IIF(:board IS NULL, "
                             "full_board_name IS NULL, "
                             "full_board_name = :board)"),
                "variable": "board",
                "visible" : True,
                },
    "_MODELS": {"endquery": "SELECT DISTINCT model_name FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE IIF(:model IS NULL, "
                             "model_name IS NULL, "
                             "model_name = :model)"),
                "variable": "model",
                "visible" : True,
                },
    "_DATES" : {"endquery": "SELECT DISTINCT DATE(created_at) FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE IIF(:date IS NULL, "
                             "created_at IS NULL, "
                             "DATE(created_at) = :date)"),
                # Maybe some way to specify a range?  Could do _AFTER and
                # _BEFORE.
                "variable": "date",
                "visible" : True,
                },
    "_BEFORE" : {"endquery": "SELECT DISTINCT DATE(created_at) FROM ({})",
                 "midquery": ("SELECT * FROM ({}) WHERE "
                              "DATE(created_at) <= :before"),
                 "variable": "before",
                 },
    "_AFTER" : {"endquery": "SELECT DISTINCT DATE(created_at) FROM ({})",
                "midquery": ("SELECT * FROM ({}) WHERE "
                             "DATE(created_at) >= :after"),
                "variable": "after",
                },
    "_IMAGES": {"endquery": "SELECT image_name FROM ({})",
                "midquery": None,
                "variable": None,
                "visible" : True,
                "consume" : [1,1],
                "final": True,
                },
    # Maybe a shorthand?
    "_I": {"endquery": "SELECT image_name FROM ({})",
           "midquery": None,
           "variable": None,
           "consume": [1,1],
           "final": True,
           },
    # I need a special case for "this filter is next-to-last"
    # which isn't working so good, and we're better without.
    "_LAST": {
        # "penultquery": ("SELECT image_name FROM ({}) "
        #                 "ORDER BY created_at DESC LIMIT :last"),
        # NO!!  This is a *grandparent* to images!
        # Can't just leave midquery blank!
        "midquery": ("SELECT * FROM ({}) ORDER BY "
                     "created_at DESC LIMIT :last"),
        "variable": "last",
        # generate_series doesn't come with the sqlite3 library...
        "endquery": ("""WITH RECURSIVE ser(x) AS (
            VALUES(10) UNION ALL SELECT x+10 FROM ser WHERE
            x<200) SELECT x FROM ser"""),
    },
    "_LIKE": {
        # Another "penultimate" one.  Less clear how to do the intermediate
        # (endquery) listing.
        # Hmm, OK, I think you have to say _LIKE/apricots/_PROMPTS which is annoying but
        # also makes sense?
        "midquery": ("SELECT * FROM ({}) WHERE positive_prompt LIKE "
                     " ('%' || :like || '%')"),
        "endquery": ("SELECT 'WORD'"), # No, you don't get hints.
        "variable": "like",
    },
    # I think... it just goes like this, easy-peasy.
    "_PROMPTS": {
        "midquery": ("SELECT * FROM ({}) WHERE "
                     "REPLACE(positive_prompt, '/', ' ') = :prompt"),
        "endquery": ("SELECT DISTINCT REPLACE(positive_prompt, '/', ' ') FROM ({})"),
        "variable": "prompt",
    }
}

ImageTbl_cmd = f"""select images.*, board_images.board_id, board_name, coalesce(board_name, '{UNSORTED}') as full_board_name, json_extract(metadata, '$.positive_prompt') as positive_prompt, coalesce(json_extract(metadata, '$.model.model_name'), models.name, json_extract(metadata, '$.model.key')) as model_name from images left join board_images on images.image_name=board_images.image_name left join boards on board_images.board_id=boards.board_id left join models on json_extract(metadata, '$.model.key')=models.id WHERE is_intermediate IS FALSE"""

ImageTbl = "all_images_boards"

class ChooseInvokeFS(fuse.Operations):
    def init(self, *args, **kwargs):
        global FILTERS
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
        self.filters = FILTERS
        if getattr(self, 'configfile', None):
            import yaml
            with open(self.configfile) as fh:
                self.filters = next(yaml.full_load_all(fh))

    def destroy(self, *args, **kwargs):
        self.cursor.close()
        self.connection.close()

    def find_filter(self, pathelts, orig_pathelts=[], filters=None):
        # Find the appropriate filter at this level.  Motivation: allow for
        # filtering that isn't alternating FILTER/value.  Will need to make
        # building the query smarter too.

        # Modifies pathelts DESTRUCTIVELY in one special case.

        if filters is None:
            filters = self.filters

        # print(f"filters: {list(filters.keys())!r}")

        # simple case:
        try:
            return filters[pathelts[0]]
        except KeyError:
            pass
        with_priority = []
        for k, v in filters.items():
            # OK, keys of the form "/str/" are regexps, anchored at both ends
            if re.match("/.*/$", k):
                if re.match(f"{k[1:-1]}$", pathelts[0]):
                    return v
            # What are some other possibilities?  Some kind of "match"
            # property in the filter that can match up other path elements?
            # Despite realizing more than once that I was building this
            # query from the top down, I *still* wrote this working on
            # previous elements as if building bottom-up, sigh.  Let's see
            # if it can be made to make sense.

            # So, a matching a PREfix of the FOLLOWING elements?  A list?
            # Or a glob-type string, maybe optionally starting with a / to
            # make it "absolute" from the mounted root.  Glob-string seems
            # more intuitive.  But probably only "*" and "**" recognized as
            # glob chars, and even those only alone (not "x*y" or
            # anything.)
            #
            # Or maybe I was on the right track in the first place, and
            # should be looking at suffixes PRECEDING elements, in which
            # case we're going to have to start passing in the "untrimmed"
            # list as well instead of just working iteratively on the list
            # being trimmed as we go.
            #
            # XXX!!! THESE STILL ARE BACKWARD I THINK.
            #
            # Really thinking matchpath really should be backward-looking,
            # so adding another param, maybe change name to backmatchpath?
            #
            # XXXXX DISABLING WITH IF False!
            if False and "matchpath" in v and orig_pathelts:
                # Using / and not os.path.sep!
                elts = v['matchpath'].split('/')
                # I'm going to say that an ending slash is optional
                if not elts[-1]:
                    elts.pop()
                def matches(m, p):
                    return (m == "*") or (m == p)
                # Not sure how to handle "**" yet.
                matched = True
                for p in reversed(orig_pathelts):
                    if not elts:
                        break
                    m = elts.pop()
                    if not matches(m, p):
                        matched = False
                        break
                if elts:
                    # everything matched so far, but if there is more to
                    # match still...
                    matched = False
                    # I think the "rooted" path will take care of itself,
                    # since both pathelts and the match elements will start
                    # with the empty string. (NO, BECAUSE THAT'S TRIMMED OFF
                    # FIRST) (BUT NOT ANYMORE?)
                if matched:
                    return v
            # Other ways to match?

            if "priority" in v:
                for p in with_priority:
                    if p["priority"] < v["priority"]:
                        with_priority.remove(p)
                # At the end, I have only things >= this one.
                # If ==, then append this one anyway, and there's
                # multiple ones.  I need only check the first, they
                # should all be the same.
                if ((not with_priority) or
                        with_priority[0]['priority'] == v['priority']):
                    with_priority.append(v)

        if len(with_priority) == 1:
            # A unique filter with highest priority
            return with_priority[0]

        # No longer chopping off top-level element anymore, so check
        # this special case for the root and  DESTRUCTIVELY pop it
        # off the pathelts, if nothing else matched.  And then
        # call recursively.
        if pathelts[0] == '' and len(pathelts) > 1:
            pathelts.pop(0)
            return self.find_filter(pathelts, orig_pathelts, filters)

        # if nothing, I mean nothing matches, I guess return something
        # vacuous.  Then it'll work for getting, say, the top-level active
        # filters at root or something, so we don't have to special-case
        # root
        return {
            "midquery": "SELECT * FROM ({})",
            "endquery": "SELECT full_board_name from ({})",
            "variable": "x",
            "consume": [1,1],
        }

    def filter_consume(self, pe, fil, query, vals, active):
        # Given a filter and path elements, consume as much of the path
        # elements (from the top!) as required by the filter (DESTRUCTIVELY
        # remove from pe param), update vals and active (DESTRUCTIVELY),
        # and return new updated query.
        #
        # "Normal" filters named by their name (which is NOT in the fil
        # param) consume up to 2 elements: one for themselves and one for
        # their value.  "Implied" ones might consume only one.  I
        # guess... let's be general. The "consume" element of the filter
        # should be a 2-element list, the first element being the minimum
        # number it consumes (there must be at least this many left), and
        # the second being the maximum?  So the default is [1,2], and
        # implied filters will be [0,1]?
        consume = fil.get("consume", [1, 2])
        eaten = []
        # Modify pe to remove them
        for _ in range(consume[0]):
            eaten.append(pe.pop(0))
            try:
                active.remove(eaten[-1])
            except ValueError:
                pass
        # In the normal case, consume[1] equals consume[0]+1.  I'm not
        # entirely certain what to do if this isn't so.
        newvals = []
        try:
            for _ in range(consume[1] - consume[0]):
                val = pe.pop(0)
                if val == NONE:
                    val = None
                newvals.append(val)
                # apply the midquery if there is more
                query = fil['midquery'].format(query)
        except IndexError:
            # Ran out of things to pop.  Apply the endquery
            ## query = fil['endquery'].format(query)
            # And empty out active.
            active.clear()
            # Should a query be "final" if endquery isn't run?  That
            # is starting to make sense, but it doesn't quite work.
        # I need to check for this ALSO, as well as the consume [1,1]?
        # apparently.  it's kludgy.
        if not fil.get("midquery", None):
            active.clear()
        if newvals and fil.get('variable', None):
            if len(newvals) == 1:
                vals[fil['variable']] = newvals[0]
            else:
                vals[fil['variable']] = newvals # what happens now???
        if fil.get("final", False):
            vals[' final '] = True
        return query

    def is_root(self, path=None, pathelts=None):
        if pathelts is None:
            pathelts = getParts(path)
        return path == os.sep or len(pathelts) == 0 or pathelts == [os.sep]

    def is_directory(self, path=None, pathelts=None):
        if not pathelts:
            pathelts=getParts(path)
        if pathelts[-1] == THISQUERY:
            return False
        # Build the query up but NOT including this
        (query, vals, active) = self.buildquery(pathelts[:-1])
        return not vals.get(' final ', False)

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

        if pe[-1] == THISQUERY:
            ss = self.thisquery(pe[:-1])
            val = ss.encode('utf8')
            st['st_mode'] = stat.S_IFREG | 0o444
            st['st_nlink'] = 1
            st['st_size'] = len(val)
            return st

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
            # Set dates on links.  I suppose could just use ls -L
            # or just go with real_files mode, which is better.
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


    def buildquery(self, pe):
        # Returns (query, vals, active); if the last is not empty
        # then we're at a "key" level and are expected to yield
        # those and not the results of the query.
        # build nested query, working from TOP DOWN, not bottom up!!.
        active = []
        for k, v in self.filters.items():
            if v.get("visible", False):
                active.append(k)
        filters = self.filters.copy() # So I can remove them as found.
        vals = {}
        query = f"SELECT * FROM {ImageTbl}" # innermost query goes to ImageTbl.
        # OK, ok, calm down.  Building from the TOP DOWN is not the same
        # process for each level.  Deciding which query of the filter to
        # use depends ONLY on the LAST element, so it must be handled
        # specially!  Higher elements should always come in pairs, and
        # we'll always use the second query!
        pec = pe.copy()
        # pe will always start with '' for the root dir.  So even numbers
        # means a key-level, odd numbers a value-level, though I will not
        # check that way...
        fil = None              # So it's available afterward
        while pec:
            fil = self.find_filter(pec, orig_pathelts=pe, filters=filters)
            if not fil:
                raise fuse.FuseOSError(errno.ENOTDIR)
            query = self.filter_consume(pec, fil, query, vals, active)
            # Also have to remove from the filters... this is the wrong
            # way to use dicts, sorry!
            for k in list(filters.keys()): # don't iterate directly.
                if filters[k] == fil:
                    # print(f"deleting filters[{k}]")
                    del filters[k]
        if not active:
            # If the last filter was non-final apply its endquery.
            # XXX!!! This should kinda be in filter_consume!!
            if fil and not vals.get(" final ", False):
                query = fil['endquery'].format(query)

        return (query, vals, active)

    def readdir(self, path, offset):
        pe = getParts(path=path)
        # print(f"{pe=}")
        if not self.is_directory(pathelts=pe):
            raise fuse.FuseOSError(errno.ENOTDIR)
        # ENOENT doesn't raise a proper error with ls, because
        # getattr() doesn't fail to find it.  To fix that,
        # getattr would have to run a db query for each entry,
        # and that slows things down.  Better just to error
        # with ENOTDIR.
        yield '.'
        yield '..'
        (query, vals, active) = self.buildquery(pe)
        final = vals.pop(" final ", False)
        if active:              # At a "key" level
            yield from iter(active)
            return
        # Is this a good idea?  I'm thinking maybe we shouldn't do this and
        # just leave an empty dir.
        if False:
            self.cursor.execute(f"SELECT COUNT(*) from ({query})", vals)
            row = self.cursor.fetchone()
            if row[0] <= 0:
                raise fuse.FuseOSError(errno.ENOTDIR)
        self.cursor.execute(query, vals)
        while (row := self.cursor.fetchone()):
            v = row[0]
            vv = str(v)
            # prompts (in particular) can be NULL *or* they can be ''.
            # We'll treat such things the same, and the config file will
            # have to have the smarts to handle it.
            if v is None or len(vv) == 0:
                yield NONE
            else:
                # OK, this actually breaks when prompt strings are too long!
                yield str(v)
        return

    def thisquery(self, pe):
        import json
        (query, vals, active) = self.buildquery(pe)
        ss = json.dumps(dict(query=query, vals=vals, active=active, pathelts=pe),
                        indent=4)
        return ss

    def read(self, path, size, offset, fh):
        # What's the FH?
        pe = getParts(path)
        # Special debugging output!
        if pe[-1] == THISQUERY:
            ss = self.thisquery(pe[:-1])
            val = ss.encode('utf8')
            return val[offset:offset+size]
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
    'configfile'  (absolute path for config .yaml file.)

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

