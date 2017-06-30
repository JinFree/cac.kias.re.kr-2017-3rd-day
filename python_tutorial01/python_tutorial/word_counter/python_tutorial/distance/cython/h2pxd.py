#!/usr/bin/env python
# Copyright (c) 2011, Evan Buswell <ebuswell@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Convert a C header file to a Cython pxd file.

This is pretty limited.  In particular, it is completely ignorant of
global variables and all effects of the C preprocessor other than
# include statements and simple (non-function) #define statements
(which are just mapped to an enum, and never evaluated).

By default, h2pxd will produce a file in a package named analogous to
the header directory (or the root package, for most headers), named as
the header but with "_c_" prepended.  So atomickit/atomic-types.h
would produce atomickit/_c_atomic_types.pxd in the output directory
(atomickit._c_atomic_types for cimport).  The file will contain a
"cdef extern from 'header':" block containing every definition in the
header file.  The "nogil" option is set by default, and currently this
behavior cannot be modified.

All options can be optionally specified in a configuration file in
addition to the command-line.  The configuration file is in standard
ini syntax, using the long option names with dashes mapped to
underscores.  The "global" section indicates global options, and in
addition each header may have its own section.  Priority of
configuration, from lowest to highest, is (1) global section of the
configuration file, (2) header section of the configuration file, (3)
command line arguments.

usage: h2pxd [-h] [-o file] [-d directory] [-I directory] [-Q] [-c file]
             [-S header] [-E header] [-Y keyword] [-D macro] [-U macro]
             header

Transform a c header file into cython pyd file.

positional arguments:
  header                C header file

optional arguments:
  -h, --help            show this help message and exit
  -o file, --out file   Output file.
  -d directory, --outdir directory
                        Directory for output file.
  -I directory, --include directory
                        Add directory to the C header search path.
  -Q, --opaque          Generate opaque structs and unions.
  -c file, --config file
                        Path to configuration file.
  -S header, --skip-include header
                        Ignore requests to include this header.
  -E header, --include-extra header
                        Add this file to the includes.
  -Y keyword, --ignore-keyword keyword
                        Ignore this keyword.
  -D macro, --define macro
                        Ignore code which depends on this macro not being
                        defined.
  -U macro, --undefine macro
                        Ignore code which depends on this macro being defined.
"""
import argparse
import re
import os
import ConfigParser
from warnings import warn

parser = argparse.ArgumentParser(description='Transform a c header file into cython pyd file.', )
parser.add_argument('-o', '--out', type=argparse.FileType('w'), metavar='file', help='Output file.')
parser.add_argument('-d', '--outdir', metavar='directory', help='Directory for output file.')
parser.add_argument('-I', '--include', action='append', metavar='directory', help='Add directory to the C header search path.')
parser.add_argument('-Q', '--opaque', action='store_true', help='Generate opaque structs and unions.')
parser.add_argument('-c', '--config', metavar='file', type=argparse.FileType('r'), help='Path to configuration file.')
parser.add_argument('-S', '--skip-include', metavar='header', action='append', help='Ignore requests to include this header.')
parser.add_argument('-E', '--include-extra', metavar='header', action='append', help='Add this file to the includes.')
parser.add_argument('-Y', '--ignore-keyword', metavar='keyword', action='append', help='Ignore this keyword.')
parser.add_argument('-D', '--define', metavar='macro', action='append', help='Ignore code which depends on this macro not being defined.')
parser.add_argument('-U', '--undefine', metavar='macro', action='append', help='Ignore code which depends on this macro being defined.')
parser.add_argument('header', metavar='header', help='C header file')
args = parser.parse_args()

if args.header[:1] == '/':
    parser.error("header '%s' should not contain an absolute path." % args.header)

if args.header[-2:] != '.h':
    parser.error("header '%s' does not appear to be the name of a C header file." % args.header)

class Config(object):
    def __init__(self, out=None, outdir=None, include=[], opaque=None, skip_include=[], include_extra=[], ignore_keyword=[], define=[], undefine=[]):
        self.out = out
        self.outdir = outdir
        self.include = include if include is not None else []
        self.opaque = opaque
        self.skip_include = skip_include if skip_include is not None else []
        self.include_extra = include_extra if include_extra is not None else []
        self.ignore_keyword = ignore_keyword if ignore_keyword is not None else []
        self.define = define if define is not None else []
        self.undefine = undefine if undefine is not None else []
    def __setattr__(self, name, value):
        if getattr(self, name, None) is None:
            self.__dict__[name] = value

conf = Config(out=args.out, outdir=args.outdir, include=args.include, opaque=args.opaque if args.opaque else None, skip_include=args.skip_include, include_extra=args.include_extra, ignore_keyword=args.ignore_keyword, define=args.define, undefine=args.undefine)

if args.config is not None:
    conffile = ConfigParser.SafeConfigParser()
    conffile.readfp(args.config, args.config.name)
    sections = []
    if conffile.has_section(args.header):
        sections.append(args.header)
    if conffile.has_section("global"):
        sections.append("global")
    for section in sections:
        for key, value in conffile.items(section):
            value = value.strip()
            if key in ('skip_include', 'include_extra', 'include', 'ignore_keyword'):
                getattr(conf, key).extend(value.split())
            elif key == 'define':
                conf.define.extend(filter(lambda x: x not in conf.undefine, value.split()))
            elif key == 'undefine':
                conf.undefine.extend(filter(lambda x: x not in conf.define, value.split()))
            elif key == 'opaque':
                if value in ('yes', 'Yes', 'y', 'Y', 'true', 'True', 't', 'T', '1'):
                    value = True
                elif value in ('no', 'No', 'n', 'N', 'false', 'False', 'f', 'F', '0'):
                    value = False
                else:
                    parser.error("opaque must be a boolean value")
                setattr(conf, key, value)
            elif key == 'out':
                value = open(value, 'r')
                setattr(conf, key, value)
            else:
                setattr(conf, key, value)

# finally, set defaults
conf.opaque = False

# first, find the header:
# in the current directory:
try:
    f = open(args.header, 'r')
except:
    f = None

if f is None:
    for directory in conf.include:
        directory = re.sub('/+', '/', directory)
        directory = re.sub('/$', '', directory)
        try:
            f = open(directory + '/' + args.header, 'r')
            break
        except:
            pass

if f is None:
    try:
        f = open('/usr/include/%s' % args.header, 'r')
    except:
        pass

if f is None:
    try:
        f = open('/usr/local/include/%s' % args.header, 'r')
    except:
        parser.error("Could not find '%s' in search path." % args.header)

if conf.out is not None:
    out = conf.out
else:
    path = args.header[:-2] + '.pxd'
    path = re.sub(r'-', '_', path)
    path = re.sub(r'(^(?:.*/)?)', r'\1_c_', path)
    base = '.'
    if conf.outdir:
        base = conf.outdir
        base = re.sub(r'/+', '/', base)
        base = re.sub(r'/$', '', base)
        if not os.path.exists(base):
            os.mkdir(base)
    full_path = base + '/' + path
    for dir in path.split('/')[:-1]:
        base += '/' + dir
        if not os.path.exists(base):
            os.mkdir(base)
            open(base + '/' + '__init__.pxd', 'w').close()
    out = open(full_path, 'w')

# ========================================================================

def skip_block(str):
    """Skips one block, taking into account nested blocks.

    Should be called with open bracket already eaten.
    """
    level = 1
    while True:
        r = re.search('^[^{}]*([{}])(.*)$', str)
        if not r:
            raise Exception("Failed to parse header: unmatched '{'")
        str = r.group(2)
        if r.group(1) == '}':
            level -= 1
            if level == 0:
                return str
        else:
            level += 1

def skip_pre_block(str):
    """Skips one preprocessor block, taking into account nested blocks.

    Should be called with open #if* already eaten.
    """
    level = 1
    while True:
        r = re.search(r'^.*?\n\s*#\s*(if|endif)(.*)$', str, re.S)
        if not r:
            raise Exception("Failed to parse header: unmatched '{'")
        str = r.group(2)
        if r.group(1) == 'endif':
            level -= 1
            if level == 0:
                return str
        else:
            level += 1

def split_statement(block):
    """Returns a tuple containing the first statement and the remainder of the block."""
    statement = ''
    while True:
        r = re.match(r'^([^{]*?) *; *(.*)$', block)
        if r:
            return (statement + r.group(1), r.group(2))
        r = re.match('^([^{]*? *{ *)(.*)$', block)
        if not r:
            return (block, '')
        statement += r.group(1)
        block = r.group(2)
        level = 1
        while True:
            r = re.search('^([^{}]* *([{}]) *)(.*)$', block)
            if not r:
                raise Exception("Failed to parse header: unmatched '{'")
            statement += r.group(1)
            block = r.group(3)
            if r.group(2) == '}':
                level -= 1
                if level == 0:
                    break
            else:
                level += 1

def parse_block(block, leader=""):
    """Converts a block to cython syntax."""
    pre = ''
    ret = leader
    if leader != '':
        inner = True
    else:
        inner = False
    indent = 8 if inner else 4
    while block:
        statement, block = split_statement(block)

        r = re.match(r'^(typedef)? *(struct|union) *([_a-zA-Z0-9]*) *((?:{ *.* *})?) *([_a-zA-Z0-9]*) *$', statement)
        if r and (r.group(1) == 'typedef' or r.group(4) != ''):
            extra = ''
            if r.group(1) == 'typedef':
                if r.group(3) != '':
                    inner_block = 4 * ' ' + 'cdef %s %s:\n' % (r.group(2), r.group(3))
                    extra = 4 * ' ' + 'ctypedef %s %s\n' % (r.group(3), r.group(5))
                else:
                    inner_block = 4 * ' ' + 'ctypedef %s %s:\n' % (r.group(2), r.group(5))
            else:
                tag = r.group(3) if r.group(3) else '_%s_%s' % (r.group(2), r.group(5))
                inner_block = 4 * ' ' + 'cdef %s %s:\n' % (r.group(2), tag)
            if conf.opaque or r.group(4) == '':
                inner_block += 8 * ' ' + 'pass\n'
            else:
                inner_block = parse_block(r.group(4)[1:-1], inner_block)
            if extra:
                inner_block += extra
            if inner:
                pre += inner_block
            else:
                ret += inner_block
            # was this a variable definition?
            if r.group(1) != 'typedef' and r.group(5) != '':
                ret += indent * ' ' + '%s %s\n' % (tag, r.group(5))
            continue

        r = re.match(r'^(typedef)? *enum *([_a-zA-Z0-9]*) *{ *(.*) *} *([_a-zA-Z0-9]*) *$', statement)
        if r:
            if r.group(1) == 'typedef':
                inner_block = 4 * ' ' + 'enum %s:\n' % r.group(4)
            else:
                tag = r.group(2) if r.group(2) else '_enum_%s' % r.group(4)
                inner_block = 4 * ' ' + 'enum %s:\n' % tag
            res = re.split(r' *, *', r.group(3))
            for item in res:
                inner_block += 8 * ' ' + item + '\n'
            if inner:
                pre += inner_block
            else:
                ret += inner_block
            # was this a variable definition?
            if r.group(1) != 'typedef' and r.group(4) != '':
                out.write(indent * ' ' + '%s %s\n' % (tag, r.group(4)))
            continue

        statement = re.sub(r'^typedef ', 'ctypedef ', statement)
        statement = re.sub(r'(?:^|(?<=[,;{}\(\)]|\s))(?:struct|enum|union)(?=[,;{}\(\)]|\s)', '', statement)
        statement = statement.strip()
        statement = re.sub(r'\s+', r' ', statement)
        statement = re.sub(r'\(\s+', r'(', statement)
        statement = re.sub(r'\(void\s*\)\s*$', r'()', statement)
        if re.match(r'^ctypedef', statement) and inner:
            pre += 4 * ' ' + statement + '\n'
        else:
            ret += indent * ' ' + statement + '\n'
    return pre + ret

def parse_include(include):
    """Translates a C include file to a Cython cinclude statement."""
    if include in ('fcnt.h', 'unistd.h'):
        include = 'posix.' + include[:-2]
    elif include in ('errno.h', 'float.h', 'limits.h', 'locale.h', 'math.h', 'signal.h', 'stddef.h', 'stdint.h', 'stdio.h', 'stdlib.h', 'string.h'):
        include = 'libc.' + include[:-2]
    else:
        include = re.sub(r'-', '_', include)
        include = re.sub(r'(^(?:.*/)?)', r'\1_c_', include)
        include = re.sub(r'/', '.', include)
        include = include[:-2]
    return "from %s cimport *\n" % include

# ==========================================================================

header = f.read()
f.close()

# remove all comments
header = re.sub(r'/\*.*?\*/', ' ', header, 0, re.S)

# get rid of quoted stuff: these could only be constants that we won't import
header = re.sub(r"'\\?.'", "''", header)
while header:
    res = re.split(r'"', header, 1)
    if len(res) == 1:
        break
    escaped = False
    end_i = -1
    for i in xrange(len(res[1])):
        if escaped:
            escaped = False
            continue
        if res[1][i] == '"':
            end_i = i + 1
            break
        if res[1][i] == "\\":
            escaped = True
    if end_i == -1:
        raise Exception("Failed to parse header: unmatched '\"'.")
    header = res[0] + "''" + res[1][end_i:]

# get rid of escaped newlines
header = re.sub(r'\\\n', ' ', header)

# get rid of preprocessor stuff about which we're explicitly informed
for define in conf.define:
    while header:
        res = re.split(r'\n\s*#\s*ifndef\s+' + re.escape(define), header, 1, re.S)
        if len(res) == 1:
            break
        header = res[0] + '\n' + skip_pre_block(res[1])

for undefine in conf.undefine:
    while header:
        res = re.split(r'\n\s*#\s*ifdef\s+' + re.escape(undefine), header, 1, re.S)
        if len(res) == 1:
            break
        header = res[0] + '\n' + skip_pre_block(res[1])

# get #includes
for include in re.finditer(r'^\s*#\s*include\s+[<"](.*)[">]\s*$', header, re.M):
    if include.group(1) in conf.skip_include:
        continue
    out.write(parse_include(include.group(1)))

for include in conf.include_extra:
    out.write(parse_include(include))

out.write("\n")

out.write('cdef extern from "%s" nogil:\n' % args.header)

# get simple #defines
for define in re.finditer(r'^\s*#\s*define\s+([a-zA-Z0-9_]+)\s+.*$', header, re.M):
    out.write(4 * ' ' + 'enum: %s\n' % define.group(1))

out.write("\n")

# remove all preprocessor directives
header = re.sub(r'^\s*#.*$', '', header, 0, re.M)

# get rid of extern, static, auto, register, const, restrict, volatile, and inline
header = re.sub(r'(?:^|(?<=[,;{}\(\)]|\s))(?:extern|static|auto|register|const|restrict|volatile|inline)(?=[,;{}\(\)]|\s)', '', header)
if conf.ignore_keyword:
    header = re.sub(r'(?:^|(?<=[,;{}\(\)]|\s))(?:' + r'|'.join([re.escape(keyword) for keyword in conf.ignore_keyword]) + r')(?=[,;{}\(\)]|\s)', '', header)

# canonicalize spaces
header = re.sub(r'([;{}])', r' \1 ', header, 0, re.S)
header = re.sub(r'\s+', ' ', header, 0, re.S)
header = header.strip()

# turn function definitions into function references
while header:
    res = re.split(r'\) { ', header, 1)
    if len(res) == 1:
        break
    header = res[0] + ') ; ' + skip_block(res[1])

# translate remainder of header

out.write(parse_block(header))
