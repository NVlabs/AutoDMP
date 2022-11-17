/*
* SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*************************************************************************
    > File Name: Msg.cpp
    > Author: Yibo Lin
    > Mail: yibolin@utexas.edu
    > Created Time: Fri 31 Jul 2015 03:20:14 PM CDT
 ************************************************************************/

#include "utility/src/msg.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

DREAMPLACE_BEGIN_NAMESPACE

const bool DREAMPLACE_DISABLE_PRINT =
    std::getenv("DREAMPLACE_DISABLE_PRINT")
        ? atoi(std::getenv("DREAMPLACE_DISABLE_PRINT"))
        : 0;

int dreamplacePrint(MessageType m, const char* format, ...) {
  if (DREAMPLACE_DISABLE_PRINT) return 0;

  va_list args;
  va_start(args, format);
  int ret = dreamplaceVPrintStream(m, stdout, format, args);
  va_end(args);

  return ret;
}

int dreamplacePrintStream(MessageType m, FILE* stream, const char* format,
                          ...) {
  va_list args;
  va_start(args, format);
  int ret = dreamplaceVPrintStream(m, stream, format, args);
  va_end(args);

  return ret;
}

int dreamplaceVPrintStream(MessageType m, FILE* stream, const char* format,
                           va_list args) {
  // print prefix
  char prefix[16];
  dreamplaceSPrintPrefix(m, prefix);
  fprintf(stream, "%s", prefix);

  // print message
  int ret = vfprintf(stream, format, args);

  return ret;
}

int dreamplaceSPrint(MessageType m, char* buf, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int ret = dreamplaceVSPrint(m, buf, format, args);
  va_end(args);

  return ret;
}

int dreamplaceVSPrint(MessageType m, char* buf, const char* format,
                      va_list args) {
  // print prefix
  char prefix[16];
  dreamplaceSPrintPrefix(m, prefix);
  sprintf(buf, "%s", prefix);

  // print message
  int ret = vsprintf(buf + strlen(prefix), format, args);

  return ret;
}

int dreamplaceSPrintPrefix(MessageType m, char* prefix) {
  switch (m) {
    case kNONE:
      return sprintf(prefix, "%c", '\0');
    case kINFO:
      return sprintf(prefix, "[INFO   ] ");
    case kWARN:
      return sprintf(prefix, "[WARNING] ");
    case kERROR:
      return sprintf(prefix, "[ERROR  ] ");
    case kDEBUG:
      return sprintf(prefix, "[DEBUG  ] ");
    case kASSERT:
      return sprintf(prefix, "[ASSERT ] ");
    default:
      dreamplaceAssertMsg(0, "unknown message type");
  }
  return 0;
}

void dreamplacePrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName,
                              const char* format, ...) {
  // construct message
  char buf[1024];
  va_list args;
  va_start(args, format);
  vsprintf(buf, format, args);
  va_end(args);

  // print message
  dreamplacePrintStream(kASSERT, stderr,
                        "%s:%u: %s: Assertion `%s' failed: %s\n", fileName,
                        lineNum, funcName, expr, buf);
}

void dreamplacePrintAssertMsg(const char* expr, const char* fileName,
                              unsigned lineNum, const char* funcName) {
  // print message
  dreamplacePrintStream(kASSERT, stderr, "%s:%u: %s: Assertion `%s' failed\n",
                        fileName, lineNum, funcName, expr);
}

DREAMPLACE_END_NAMESPACE
