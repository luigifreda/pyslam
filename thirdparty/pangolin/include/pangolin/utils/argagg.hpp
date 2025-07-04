/*
 * @file
 * @brief
 * Defines a very simple command line argument parser.
 *
 * @copyright
 * Copyright (c) 2017 Viet The Nguyen
 *
 * @copyright
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * @copyright
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * @copyright
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#pragma once
#ifndef ARGAGG_ARGAGG_ARGAGG_HPP
#define ARGAGG_ARGAGG_ARGAGG_HPP

#ifdef __unix__
#include <stdio.h>
#include <unistd.h>
#endif // #ifdef __unix__

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <iterator>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


/**
 * @brief
 * There are only two hard things in Computer Science: cache invalidation and
 * naming things (Phil Karlton).
 *
 * The names of types have to be succint and clear. This has turned out to be a
 * more difficult thing than I expected. Here you'll find a quick overview of
 * the type names you'll find in this namespace (and thus "library").
 *
 * When a program is invoked it is passed a number of "command line arguments".
 * Each of these "arguments" is a string (C-string to be more precise). An
 * "option" is a command line argument that has special meaning. This library
 * recognizes a command line argument as a potential option if it starts with a
 * dash ('-') or double-dash ('--').
 *
 * A "parser" is a set of "definitions" (not a literal std::set but rather a
 * std::vector). A parser is represented by the argagg::parser struct.
 *
 * A "definition" is a structure with four components that define what
 * "options" are recognized. The four components are the name of the option,
 * the strings that represent the option, the option's help text, and how many
 * arguments the option should expect. "Flags" are the individual strings that
 * represent the option ("-v" and "--verbose" are flags for the "verbose"
 * option). A definition is represented by the argagg::definition struct.
 *
 * Note at this point that the word "option" can be used interchangeably to
 * mean the notion of an option and the actual instance of an option given a
 * set of command line arguments. To be unambiguous we use a "definition" to
 * represent the notion of an option and an "option result" to represent an
 * actual option parsed from a set of command line arguments. An "option
 * result" is represented by the argagg::option_result struct.
 *
 * There's one more wrinkle to this: an option can show up multiple times in a
 * given set of command line arguments. For example, "-n 1 -n 2 -n 3". This
 * will parse into three distinct argagg::option_result instances, but all of
 * them correspond to the same argagg::definition. We aggregate these into the
 * argagg::option_results struct which represents "all parser results for a
 * given option definition". This argagg::option_results is basically a
 * std::vector of argagg::option_result.
 *
 * Options aren't the only thing parsed though. Positional arguments are also
 * parsed. Thus a parser produces a result that contains both option results
 * and positional arguments. The parser results are represented by the
 * argagg::parser_results struct. All option results are stored in a mapping
 * from option name to the argagg::option_results. All positional arguments are
 * simply stored in a vector of C-strings.
 */
namespace argagg {


/**
 * @brief
 * This exception is thrown when a long option is parsed and is given an
 * argument using the "=" syntax but the option doesn't expect an argument.
 */
struct unexpected_argument_error
: public std::runtime_error {
  using std::runtime_error::runtime_error;
};


/**
 * @brief
 * This exception is thrown when an option is parsed unexpectedly such as when
 * an argument was expected for a previous option or if an option was found
 * that has not been defined.
 */
struct unexpected_option_error
: public std::runtime_error {
  using std::runtime_error::runtime_error;
};


/**
 * @brief
 * This exception is thrown when an option requires an argument but is not
 * provided one. This can happen if another flag was found after the option or
 * if we simply reach the end of the command line arguments.
 */
struct option_lacks_argument_error
: public std::runtime_error {
  using std::runtime_error::runtime_error;
};


/**
 * @brief
 * This exception is thrown when an option's flag is invalid. This can be the
 * case if the flag is not prefixed by one or two hyphens or contains non
 * alpha-numeric characters after the hypens. See is_valid_flag_definition()
 * for more details.
 */
struct invalid_flag
: public std::runtime_error {
  using std::runtime_error::runtime_error;
};


/**
 * @brief
 * The set of template instantiations that convert C-strings to other types for
 * the option_result::as(), option_results::as(), parser_results::as(), and
 * parser_results::all_as() methods are placed in this namespace.
 */
namespace convert {

  /**
   * @brief
   * Explicit instantiations of this function are used to convert arguments to
   * types.
   */
  template <typename T>
  T arg(const char* arg);

}


/**
 * @brief
 * Represents a single option parse result.
 *
 * You can check if this has an argument by using the implicit boolean
 * conversion.
 */
struct option_result {

  /**
   * @brief
   * Argument parsed for this single option. If no argument was parsed this
   * will be set to nullptr.
   */
  const char* arg;

  /**
   * @brief
   * Converts the argument parsed for this single option instance into the
   * given type using the type matched conversion function
   * argagg::convert::arg(). If there was not an argument parsed for this
   * single option instance then a argagg::option_lacks_argument_error
   * exception is thrown. The specific conversion function may throw other
   * exceptions.
   */
  template <typename T>
  T as() const;

  /**
   * @brief
   * Converts the argument parsed for this single option instance into the
   * given type using the type matched conversion function
   * argagg::convert::arg(). If there was not an argument parsed for this
   * single option instance then the provided default value is returned
   * instead. If the conversion function throws an exception then it is ignored
   * and the default value is returned.
   */
  template <typename T>
  T as(const T& t) const;

  /**
   * @brief
   * Since we have the argagg::option_result::as() API we might as well alias
   * it as an implicit conversion operator. This performs implicit conversion
   * using the argagg::option_result::as() method.
   *
   * @note
   * An implicit boolean conversion specialization exists which returns false
   * if there is no argument for this single option instance and true
   * otherwise. This specialization DOES NOT convert the argument to a bool. If
   * you need to convert the argument to a bool then use the as() API.
   */
  template <typename T>
  operator T () const;

};


/**
 * @brief
 * Represents multiple option parse results for a single option. If treated as
 * a single parse result it defaults to the last parse result. Note that an
 * instance of this struct is always created even if no option results are
 * parsed for a given definition. In that case it will simply be empty.
 *
 * To check if the associated option showed up at all simply use the implicit
 * boolean conversion or check if count() is greater than zero.
 */
struct option_results {

  /**
   * @brief
   * All option parse results for this option.
   */
  std::vector<option_result> all;

  /**
   * @brief
   * Gets the number of times the option shows up.
   */
  std::size_t count() const;

  /**
   * @brief
   * Gets a single option parse result by index.
   */
  option_result& operator [] (std::size_t index);

  /**
   * @brief
   * Gets a single option result by index.
   */
  const option_result& operator [] (std::size_t index) const;

  /**
   * @brief
   * Converts the argument parsed for the LAST option parse result for the
   * parent definition to the provided type. For example, if this was for "-f 1
   * -f 2 -f 3" then calling this method for an integer type will return 3. If
   * there are no option parse results then a std::out_of_range exception is
   * thrown. Any exceptions thrown by option_result::as() are not
   * handled.
   */
  template <typename T>
  T as() const;

  /**
   * @brief
   * Converts the argument parsed for the LAST option parse result for the
   * parent definition to the provided type. For example, if this was for "-f 1
   * -f 2 -f 3" then calling this method for an integer type will return 3. If
   * there are no option parse results then the provided default value is
   * returned instead.
   */
  template <typename T>
  T as(const T& t) const;

  /**
   * @brief
   * Since we have the option_results::as() API we might as well alias
   * it as an implicit conversion operator. This performs implicit conversion
   * using the option_results::as() method.
   *
   * @note
   * An implicit boolean conversion specialization exists which returns false
   * if there is no argument for this single option instance and true
   * otherwise. This specialization DOES NOT convert the argument to a bool. If
   * you need to convert the argument to a bool then use the as() API.
   */
  template <typename T>
  operator T () const;

};


/**
 * @brief
 * Represents all results of the parser including options and positional
 * arguments.
 */
struct parser_results {

  /**
   * @brief
   * Returns the name of the program from the original arguments list. This is
   * always the first argument.
   */
  const char* program;

  /**
   * @brief
   * Maps from definition name to the structure which contains the parser
   * results for that definition.
   */
  std::unordered_map<std::string, option_results> options;

  /**
   * @brief
   * Vector of positional arguments.
   */
  std::vector<const char*> pos;

  /**
   * @brief
   * Used to check if an option was specified at all.
   */
  bool has_option(const std::string& name) const;

  /**
   * @brief
   * Get the parser results for the given definition. If the definition never
   * showed up then the exception from the unordered_map access will bubble
   * through so check if the flag exists in the first place with has_option().
   */
  option_results& operator [] (const std::string& name);

  /**
   * @brief
   * Get the parser results for the given definition. If the definition never
   * showed up then the exception from the unordered_map access will bubble
   * through so check if the flag exists in the first place with has_option().
   */
  const option_results& operator [] (const std::string& name) const;

  /**
   * @brief
   * Gets the number of positional arguments.
   */
  std::size_t count() const;

  /**
   * @brief
   * Gets a positional argument by index.
   */
  const char* operator [] (std::size_t index) const;

  /**
   * @brief
   * Gets a positional argument converted to the given type.
   */
  template <typename T>
  T as(std::size_t i = 0) const;

  /**
   * @brief
   * Gets all positional arguments converted to the given type.
   */
  template <typename T>
  std::vector<T> all_as() const;

};


/**
 * @brief
 * An option definition which essentially represents what an option is.
 */
struct definition {

  /**
   * @brief
   * Name of the option. Option parser results are keyed by this name.
   */
  const std::string name;

  /**
   * @brief
   * List of strings to match that correspond to this option. Should be fully
   * specified with hyphens (e.g. "-v" or "--verbose").
   */
  std::vector<std::string> flags;

  /**
   * @brief
   * Help string for this option.
   */
  std::string help;

  /**
   * @brief
   * Number of arguments this option requires. Must be 0 or 1. All other values
   * have undefined behavior. Okay, the code actually works with positive
   * values in general, but it's unorthodox command line behavior.
   */
  unsigned int num_args;

  /**
   * @brief
   * Returns true if this option does not want any arguments.
   */
  bool wants_no_arguments() const;

  /**
   * @brief
   * Returns true if this option requires arguments.
   */
  bool requires_arguments() const;

};


/**
 * @brief
 * Checks whether or not a command line argument should be processed as an
 * option flag. This is very similar to is_valid_flag_definition() but must
 * allow for short flag groups (e.g. "-abc") and equal-assigned long flag
 * arguments (e.g. "--output=foo.txt").
 */
bool cmd_line_arg_is_option_flag(
  const char* s);


/**
 * @brief
 * Checks whether a flag in an option definition is valid. I suggest reading
 * through the function source to understand what dictates a valid.
 */
bool is_valid_flag_definition(
  const char* s);


/**
 * @brief
 * Tests whether or not a valid flag is short. Assumes the provided cstring is
 * already a valid flag.
 */
bool flag_is_short(
  const char* s);


/**
 * @brief
 * Contains two maps which aid in option parsing. The first map, @ref
 * short_map, maps from a short flag (just a character) to a pointer to the
 * original @ref definition that the flag represents. The second map, @ref
 * long_map, maps from a long flag (an std::string) to a pointer to the
 * original @ref definition that the flag represents.
 *
 * This object is usually a temporary that only exists during the parsing
 * operation. It is typically constructed using @ref validate_definitions().
 */
struct parser_map {

  /**
   * @brief
   * Maps from a short flag (just a character) to a pointer to the original
   * @ref definition that the flag represents.
   */
  std::array<const definition*, 256> short_map;

  /**
   * @brief
   * Maps from a long flag (an std::string) to a pointer to the original @ref
   * definition that the flag represents.
   */
  std::unordered_map<std::string, const definition*> long_map;

  /**
   * @brief
   * Returns true if the provided short flag exists in the map object.
   */
  bool known_short_flag(
    const char flag) const;

  /**
   * @brief
   * If the short flag exists in the map object then it is returned by this
   * method. If it doesn't then nullptr will be returned.
   */
  const definition* get_definition_for_short_flag(
    const char flag) const;

  /**
   * @brief
   * Returns true if the provided long flag exists in the map object.
   */
  bool known_long_flag(
    const std::string& flag) const;

  /**
   * @brief
   * If the long flag exists in the map object then it is returned by this
   * method. If it doesn't then nullptr will be returned.
   */
  const definition* get_definition_for_long_flag(
    const std::string& flag) const;

};


/**
 * @brief
 * Validates a collection (specifically an std::vector) of @ref definition
 * objects by checking if the contained flags are valid. If the set of @ref
 * definition objects is not valid then an exception is thrown. Upon successful
 * validation a @ref parser_map object is returned.
 */
parser_map validate_definitions(
  const std::vector<definition>& definitions);


/**
 * @brief
 * A list of option definitions used to inform how to parse arguments.
 */
struct parser {

  /**
   * @brief
   * Vector of the option definitions which inform this parser how to parse
   * the command line arguments.
   */
  std::vector<definition> definitions;

  /**
   * @brief
   * Parses the provided command line arguments and returns the results as
   * @ref parser_results.
   *
   * @note
   * This method is not thread-safe and assumes that no modifications are made
   * to the definitions member field during the extent of this method call.
   */
  parser_results parse(int argc, const char** argv) const;

  /**
   * @brief
   * Through strict interpretation of pointer casting rules, despite this being
   * a safe operation, C++ doesn't allow implicit casts from <tt>char**</tt> to
   * <tt>const char**</tt> so here's an overload that performs a const_cast,
   * which is typically frowned upon but is safe here.
   */
  parser_results parse(int argc, char** argv) const;

};


/**
 * @brief
 * A convenience output stream that will accumulate what is streamed to it and
 * then, on destruction, format the accumulated string using the fmt program
 * (via the argagg::fmt_string() function) to the provided std::ostream.
 *
 * Example use:
 *
 * @code
 * {
 *   argagg::fmt_ostream f(std::cerr);
 *   f << "Usage: " << really_long_string << std::endl;
 * } // on destruction here the formatted string will be streamed to std::cerr
 * @endcode
 *
 * @note
 * This only has formatting behavior if the <tt>__unix__</tt> preprocessor
 * definition is defined since formatting relies on the POSIX API for forking,
 * executing a process, and reading/writing to/from file descriptors. If that
 * preprocessor definition is not defined then this class has the same overall
 * behavior except the output string is not formatted (basically streams
 * whatever the accumulated string is). See arggg::fmt_string().
 */
struct fmt_ostream : public std::ostringstream {

  /**
   * @brief
   * Reference to the final output stream that the formatted string will be
   * streamed to.
   */
  std::ostream& output;

  /**
   * @brief
   * Construct to output to the provided output stream when this object is
   * destroyed.
   */
  fmt_ostream(std::ostream& output);

  /**
   * @brief
   * Special destructor that will format the accumulated string using fmt (via
   * the argagg::fmt_string() function) and stream it to the std::ostream
   * stored.
   */
  ~fmt_ostream();

};


/**
 * @brief
 * Processes the provided string using the fmt util and returns the resulting
 * output as a string. Not the most efficient (in time or space) but gets the
 * job done.
 *
 * This function is cowardly so if there are any errors encountered such as a
 * syscall returning -1 then the input string is returned.
 *
 * @note
 * This only has formatting behavior if the <tt>__unix__</tt> preprocessor
 * definition is defined since it relies on the POSIX API for forking,
 * executing a process, reading/writing to/from file descriptors, and the
 * existence of the fmt util.
 */
std::string fmt_string(const std::string& s);


} // namespace argagg


/**
 * @brief
 * Writes the option help to the given stream.
 */
std::ostream& operator << (std::ostream& os, const argagg::parser& x);


// ---- end of declarations, header-only implementations follow ----


namespace argagg {


template <typename T>
T option_result::as() const
{
  if (this->arg) {
    return convert::arg<T>(this->arg);
  } else {
    throw option_lacks_argument_error("option has no argument");
  }
}


template <typename T>
T option_result::as(const T& t) const
{
  if (this->arg) {
    try {
      return convert::arg<T>(this->arg);
    } catch (...) {
      return t;
    }
  } else {
    // I actually think this will never happen. To call this method you have
    // to access a specific option_result for an option. If there's a
    // specific option_result then the option was found. If the option
    // requires an argument then it will definitely have an argument
    // otherwise the parser would have complained.
    return t;
  }
}


template <typename T>
option_result::operator T () const
{
  return this->as<T>();
}


template <> inline
option_result::operator bool () const
{
  return this->arg != nullptr;
}


inline
std::size_t option_results::count() const
{
  return this->all.size();
}


inline
option_result& option_results::operator [] (std::size_t index)
{
  return this->all[index];
}


inline
const option_result& option_results::operator [] (std::size_t index) const
{
  return this->all[index];
}


template <typename T>
T option_results::as() const
{
  if (this->all.size() == 0) {
    throw std::out_of_range("no option arguments to convert");
  }
  return this->all.back().as<T>();
}


template <typename T>
T option_results::as(const T& t) const
{
  if (this->all.size() == 0) {
    return t;
  }
  return this->all.back().as<T>(t);
}


template <typename T>
option_results::operator T () const
{
  return this->as<T>();
}


template <> inline
option_results::operator bool () const
{
  return this->all.size() > 0;
}


inline
bool parser_results::has_option(const std::string& name) const
{
  const auto it = this->options.find(name);
  return ( it != this->options.end()) && it->second.all.size() > 0;
}


inline
option_results& parser_results::operator [] (const std::string& name)
{
  return this->options.at(name);
}


inline
const option_results&
parser_results::operator [] (const std::string& name) const
{
  return this->options.at(name);
}


inline
std::size_t parser_results::count() const
{
  return this->pos.size();
}


inline
const char* parser_results::operator [] (std::size_t index) const
{
  return this->pos[index];
}


template <typename T>
T parser_results::as(std::size_t i) const
{
  return convert::arg<T>(this->pos[i]);
}


template <typename T>
std::vector<T> parser_results::all_as() const
{
  std::vector<T> v(this->pos.size());
  std::transform(
    this->pos.begin(), this->pos.end(), v.begin(),
    [](const char* arg) {
      return convert::arg<T>(arg);
    });
  return v;
}


inline
bool definition::wants_no_arguments() const
{
  return this->num_args == 0;
}


inline
bool definition::requires_arguments() const
{
  return this->num_args > 0;
}


inline
bool cmd_line_arg_is_option_flag(
  const char* s)
{
  auto len = std::strlen(s);

  // The shortest possible flag has two characters: a hyphen and an
  // alpha-numeric character.
  if (len < 2) {
    return false;
  }

  // All flags must start with a hyphen.
  if (s[0] != '-') {
    return false;
  }

  // Shift the name forward by a character to account for the initial hyphen.
  // This means if s was originally "-v" then name will be "v".
  const char* name = s + 1;

  // Check if we're dealing with a long flag.
  bool is_long = false;
  if (s[1] == '-') {
    is_long = true;

    // Just -- is not a valid flag.
    if (len == 2) {
      return false;
    }

    // Shift the name forward to account for the extra hyphen. This means if s
    // was originally "--output" then name will be "output".
    name = s + 2;
  }

  // The first character of the flag name must be alpha-numeric. This is to
  // prevent things like "---a" from being valid flags.
  len = std::strlen(name);
  if (!std::isalnum(name[0])) {
    return false;
  }

  // At this point in is_valid_flag_definition() we would check if the short
  // flag has only one character. At command line specification you can group
  // short flags together or even add an argument to a short flag without a
  // space delimiter. Thus we don't check if this has only one character
  // because it might not.

  // If this is a long flag then we expect all characters *up to* an equal sign
  // to be alpha-numeric or a hyphen. After the equal sign you are specify the
  // argument to a long flag which can be basically anything.
  if (is_long) {
    bool encountered_equal = false;
    return std::all_of(name, name + len, [&](const char& c) {
        if (encountered_equal) {
          return true;
        } else {
          if (c == '=') {
            encountered_equal = true;
            return true;
          }
          return std::isalnum(c) || c == '-';
        }
      });
  }

  // At this point we are not dealing with a long flag. We already checked that
  // the first character is alpha-numeric so we've got the case of a single
  // short flag covered. This might be a short flag group though and we might
  // be tempted to check that each character of the short flag group is
  // alpha-numeric. However, you can specify the argument for a short flag
  // without a space delimiter (e.g. "-I/usr/local/include") so you can't tell
  // if the rest of a short flag group is part of the argument or not unless
  // you know what is a defined flag or not. We leave that kind of processing
  // to the parser.
  return true;
}


inline
bool is_valid_flag_definition(
  const char* s)
{
  auto len = std::strlen(s);

  // The shortest possible flag has two characters: a hyphen and an
  // alpha-numeric character.
  if (len < 2) {
    return false;
  }

  // All flags must start with a hyphen.
  if (s[0] != '-') {
    return false;
  }

  // Shift the name forward by a character to account for the initial hyphen.
  // This means if s was originally "-v" then name will be "v".
  const char* name = s + 1;

  // Check if we're dealing with a long flag.
  bool is_long = false;
  if (s[1] == '-') {
    is_long = true;

    // Just -- is not a valid flag.
    if (len == 2) {
      return false;
    }

    // Shift the name forward to account for the extra hyphen. This means if s
    // was originally "--output" then name will be "output".
    name = s + 2;
  }

  // The first character of the flag name must be alpha-numeric. This is to
  // prevent things like "---a" from being valid flags.
  len = std::strlen(name);
  if (!std::isalnum(name[0])) {
    return false;
  }

  // If this is a short flag then it must only have one character.
  if (!is_long && len > 1) {
    return false;
  }

  // The rest of the characters must be alpha-numeric, but long flags are
  // allowed to have hyphens too.
  return std::all_of(name + 1, name + len, [&](const char& c) {
      return std::isalnum(c) || (c == '-' && is_long);
    });
}


inline
bool flag_is_short(
  const char* s)
{
  return s[0] == '-' && std::isalnum(s[1]);
}


inline
bool parser_map::known_short_flag(
  const char flag) const
{
  return this->short_map[flag] != nullptr;
}


inline
const definition* parser_map::get_definition_for_short_flag(
  const char flag) const
{
  return this->short_map[flag];
}


inline
bool parser_map::known_long_flag(
  const std::string& flag) const
{
  const auto existing_long_flag = this->long_map.find(flag);
  return existing_long_flag != long_map.end();
}


inline
const definition* parser_map::get_definition_for_long_flag(
  const std::string& flag) const
{
  const auto existing_long_flag = this->long_map.find(flag);
  if (existing_long_flag == long_map.end()) {
    return nullptr;
  }
  return existing_long_flag->second;
}


inline
parser_map validate_definitions(
  const std::vector<definition>& definitions)
{
  std::unordered_map<std::string, const definition*> long_map;
  parser_map map {{{nullptr}}, std::move(long_map)};

  for (auto& defn : definitions) {

    if (defn.flags.size() == 0) {
      std::ostringstream msg;
      msg << "option \"" << defn.name << "\" has no flag definitions";
      throw invalid_flag(msg.str());
    }

    for (auto& flag : defn.flags) {

      if (!is_valid_flag_definition(flag.data())) {
        std::ostringstream msg;
        msg << "flag \"" << flag << "\" specified for option \"" << defn.name
            << "\" is invalid";
        throw invalid_flag(msg.str());
      }

      if (flag_is_short(flag.data())) {
        const int short_flag_letter = flag[1];
        const auto existing_short_flag = map.short_map[short_flag_letter];
        bool short_flag_already_exists = (existing_short_flag != nullptr);
        if (short_flag_already_exists) {
          std::ostringstream msg;
          msg << "duplicate short flag \"" << flag
              << "\" found, specified by both option  \"" << defn.name
              << "\" and option \"" << existing_short_flag->name;
          throw invalid_flag(msg.str());
        }
        map.short_map[short_flag_letter] = &defn;
        continue;
      }

      // If we're here then this is a valid, long-style flag.
      if (map.known_long_flag(flag)) {
        const auto existing_long_flag = map.get_definition_for_long_flag(flag);
        std::ostringstream msg;
        msg << "duplicate long flag \"" << flag
            << "\" found, specified by both option  \"" << defn.name
            << "\" and option \"" << existing_long_flag->name;
        throw invalid_flag(msg.str());
      }
      map.long_map.insert(std::make_pair(flag, &defn));
    }
  }

  return map;
}


inline
parser_results parser::parse(int argc, const char** argv) const
{
  // Inspect each definition to see if its valid. You may wonder "why don't
  // you do this validation on construction?" I had thought about it but
  // realized that since I've made the parser an aggregate type (granted it
  // just "aggregates" a single vector) I would need to track any changes to
  // the definitions vector and re-run the validity check in order to
  // maintain this expected "validity invariant" on the object. That would
  // then require hiding the definitions vector as a private entry and then
  // turning the parser into a thin interface (by re-exposing setters and
  // getters) to the vector methods just so that I can catch when the
  // definition has been modified. It seems much simpler to just enforce the
  // validity when you actually want to parser because it's at the moment of
  // parsing that you know the definitions are complete.
  parser_map map = validate_definitions(this->definitions);

  // Initialize the parser results that we'll be returning. Store the program
  // name (assumed to be the first command line argument) and initialize
  // everything else as empty.
  std::unordered_map<std::string, option_results> options {};
  std::vector<const char*> pos;
  parser_results results {argv[0], std::move(options), std::move(pos)};

  // Add an empty option result for each definition.
  for (const auto& defn : this->definitions) {
    option_results opt_results {{}};
    results.options.insert(
      std::make_pair(defn.name, opt_results));
  }

  // Don't start off ignoring flags. We only ignore flags after a -- shows up
  // in the command line arguments.
  bool ignore_flags = false;

  // Keep track of any options that are expecting arguments.
  const char* last_flag_expecting_args = nullptr;
  option_result* last_option_expecting_args = nullptr;
  unsigned int num_option_args_to_consume = 0;

  // Get pointers to pointers so we can treat the raw pointer array as an
  // iterator for standard library algorithms. This isn't used yet but can be
  // used to template this function to work on iterators over strings or
  // C-strings.
  const char** arg_i = argv + 1;
  const char** arg_end = argv + argc;

  while (arg_i != arg_end) {
    auto arg_i_cstr = *arg_i;
    auto arg_i_len = std::strlen(arg_i_cstr);

    // Some behavior to note: if the previous option is expecting an argument
    // then the next entry will be treated as a positional argument even if
    // it looks like a flag.
    bool treat_as_positional_argument = (
        ignore_flags
        || num_option_args_to_consume > 0
        || !cmd_line_arg_is_option_flag(arg_i_cstr)
      );
    if (treat_as_positional_argument) {

      // If last option is expecting some specific positive number of
      // arguments then give this argument to that option, *regardless of
      // whether or not the argument looks like a flag or is the special "--"
      // argument*.
      if (num_option_args_to_consume > 0) {
        last_option_expecting_args->arg = arg_i_cstr;
        --num_option_args_to_consume;
        ++arg_i;
        continue;
      }

      // Now we check if this is just "--" which is a special argument that
      // causes all following arguments to be treated as non-options and is
      // itselve discarded.
      if (std::strncmp(arg_i_cstr, "--", 2) == 0 && arg_i_len == 2) {
        ignore_flags = true;
        ++arg_i;
        continue;
      }

      // If there are no expectations for option arguments then simply use
      // this argument as a positional argument.
      results.pos.push_back(arg_i_cstr);
      ++arg_i;
      continue;
    }

    // Reset the "expecting argument" state.
    last_flag_expecting_args = nullptr;
    last_option_expecting_args = nullptr;
    num_option_args_to_consume = 0;

    // If we're at this point then we're definitely dealing with something
    // that is flag-like and has hyphen as the first character and has a
    // length of at least two characters. How we handle this potential flag
    // depends on whether or not it is a long-option so we check that first.
    bool is_long_flag = (arg_i_cstr[1] == '-');

    if (is_long_flag) {

      // Long flags have a complication: their arguments can be specified
      // using an '=' character right inside the argument. That means an
      // argument like "--output=foobar.txt" is actually an option with flag
      // "--output" and argument "foobar.txt". So we look for the first
      // instance of the '=' character and keep it in long_flag_arg. If
      // long_flag_arg is nullptr then we didn't find '='. We need the
      // flag_len to construct long_flag_str below.
      auto long_flag_arg = std::strchr(arg_i_cstr, '=');
      std::size_t flag_len = arg_i_len;
      if (long_flag_arg != nullptr) {
        flag_len = long_flag_arg - arg_i_cstr;
      }
      std::string long_flag_str(arg_i_cstr, flag_len);

      if (!map.known_long_flag(long_flag_str)) {
        std::ostringstream msg;
        msg << "found unexpected flag: " << long_flag_str;
        throw unexpected_option_error(msg.str());
      }

      const auto defn = map.get_definition_for_long_flag(long_flag_str);

      if (long_flag_arg != nullptr && defn->num_args == 0) {
        std::ostringstream msg;
        msg << "found argument for option not expecting an argument: "
            << arg_i_cstr;
        throw unexpected_argument_error(msg.str());
      }

      // We've got a legitimate, known long flag option so we add an option
      // result. This option result initially has an arg of nullptr, but that
      // might change in the following block.
      auto& opt_results = results.options[defn->name];
      option_result opt_result {nullptr};
      opt_results.all.push_back(std::move(opt_result));

      if (defn->requires_arguments()) {
        bool there_is_an_equal_delimited_arg = (long_flag_arg != nullptr);
        if (there_is_an_equal_delimited_arg) {
          // long_flag_arg would be "=foo" in the "--output=foo" case so we
          // increment by 1 to get rid of the equal sign.
          opt_results.all.back().arg = long_flag_arg + 1;
        } else {
          last_flag_expecting_args = arg_i_cstr;
          last_option_expecting_args = &(opt_results.all.back());
          num_option_args_to_consume = defn->num_args;
        }
      }

      ++arg_i;
      continue;
    }

    // If we've made it here then we're looking at either a short flag or a
    // group of short flags. Short flags can be grouped together so long as
    // they don't require any arguments unless the option that does is the
    // last in the group ("-o x -v" is okay, "-vo x" is okay, "-ov x" is
    // not). So starting after the dash we're going to process each character
    // as if it were a separate flag. Note "sf_idx" stands for "short flag
    // index".
    for (std::size_t sf_idx = 1; sf_idx < arg_i_len; ++sf_idx) {
      const auto short_flag = arg_i_cstr[sf_idx];

      if (!std::isalnum(short_flag)) {
        std::ostringstream msg;
        msg << "found non-alphanumeric character '" << arg_i_cstr[sf_idx]
            << "' in flag group '" << arg_i_cstr << "'";
        throw std::domain_error(msg.str());
      }

      if (!map.known_short_flag(short_flag)) {
        std::ostringstream msg;
        msg << "found unexpected flag '" << arg_i_cstr[sf_idx]
            << "' in flag group '" << arg_i_cstr << "'";
        throw unexpected_option_error(msg.str());
      }

      auto defn = map.get_definition_for_short_flag(short_flag);
      auto& opt_results = results.options[defn->name];

      // Create an option result with an empty argument (for now) and add it
      // to this option's results.
      option_result opt_result {nullptr};
      opt_results.all.push_back(std::move(opt_result));

      if (defn->requires_arguments()) {

        // If this short flag's option requires an argument and we're the
        // last flag in the short flag group then just put the parser into
        // "expecting argument for last option" state and move onto the next
        // command line argument.
        bool is_last_short_flag_in_group = (sf_idx == arg_i_len - 1);
        if (is_last_short_flag_in_group) {
          last_flag_expecting_args = arg_i_cstr;
          last_option_expecting_args = &(opt_results.all.back());
          num_option_args_to_consume = defn->num_args;
          break;
        }

        // If this short flag's option requires an argument and we're NOT the
        // last flag in the short flag group then we automatically consume
        // the rest of the short flag group as the argument for this flag.
        // This is how we get the POSIX behavior of being able to specify a
        // flag's arguments without a white space delimiter (e.g.
        // "-I/usr/local/include").
        opt_results.all.back().arg = arg_i_cstr + sf_idx + 1;
        break;
      }
    }

    ++arg_i;
    continue;
  }

  // If we're done with all of the arguments but are still expecting
  // arguments for a previous option then we haven't satisfied that option.
  // This is an error.
  if (num_option_args_to_consume > 0) {
    std::ostringstream msg;
    msg << "last option \"" << last_flag_expecting_args
        << "\" expects an argument but the parser ran out of command line "
        << "arguments to parse";
    throw option_lacks_argument_error(msg.str());
  }

  return results;
}


inline
parser_results parser::parse(int argc, char** argv) const
{
  return parse(argc, const_cast<const char**>(argv));
}


namespace convert {


  /**
   * @brief
   * Templated function for conversion to T using the @ref std::strtol()
   * function.  This is used for anything long length or shorter (long, int,
   * short, char).
   */
  template <typename T> inline
  T long_(const char* arg)
  {
    char* endptr = nullptr;
    errno = 0;
    T ret = static_cast<T>(std::strtol(arg, &endptr, 0));
    if (endptr == arg) {
      std::ostringstream msg;
      msg << "unable to convert argument to integer: \"" << arg << "\"";
      throw std::invalid_argument(msg.str());
    }
    if (errno == ERANGE) {
      throw std::out_of_range("argument numeric value out of range");
    }
    return ret;
  }


  /**
   * @brief
   * Templated function for conversion to T using the @ref std::strtoll()
   * function.  This is used for anything long long length or shorter (long
   * long).
   */
  template <typename T> inline
  T long_long_(const char* arg)
  {
    char* endptr = nullptr;
    errno = 0;
    T ret = static_cast<T>(std::strtoll(arg, &endptr, 0));
    if (endptr == arg) {
      std::ostringstream msg;
      msg << "unable to convert argument to integer: \"" << arg << "\"";
      throw std::invalid_argument(msg.str());
    }
    if (errno == ERANGE) {
      throw std::out_of_range("argument numeric value out of range");
    }
    return ret;
  }


#define DEFINE_CONVERSION_FROM_LONG_(TYPE) \
  template <> inline \
  TYPE arg(const char* arg) \
  { \
    return long_<TYPE>(arg); \
  }

  DEFINE_CONVERSION_FROM_LONG_(char)
  DEFINE_CONVERSION_FROM_LONG_(unsigned char)
  DEFINE_CONVERSION_FROM_LONG_(signed char)
  DEFINE_CONVERSION_FROM_LONG_(short)
  DEFINE_CONVERSION_FROM_LONG_(unsigned short)
  DEFINE_CONVERSION_FROM_LONG_(int)
  DEFINE_CONVERSION_FROM_LONG_(unsigned int)
  DEFINE_CONVERSION_FROM_LONG_(long)
  DEFINE_CONVERSION_FROM_LONG_(unsigned long)

#undef DEFINE_CONVERSION_FROM_LONG_


#define DEFINE_CONVERSION_FROM_LONG_LONG_(TYPE) \
  template <> inline \
  TYPE arg(const char* arg) \
  { \
    return long_long_<TYPE>(arg); \
  }

  DEFINE_CONVERSION_FROM_LONG_LONG_(long long)
  DEFINE_CONVERSION_FROM_LONG_LONG_(unsigned long long)

#undef DEFINE_CONVERSION_FROM_LONG_LONG_


  template <> inline
  bool arg(const char* arg)
  {
    return argagg::convert::arg<int>(arg) != 0;
  }


  template <> inline
  float arg(const char* arg)
  {
    char* endptr = nullptr;
    errno = 0;
    float ret = std::strtof(arg, &endptr);
    if (endptr == arg) {
      std::ostringstream msg;
      msg << "unable to convert argument to integer: \"" << arg << "\"";
      throw std::invalid_argument(msg.str());
    }
    if (errno == ERANGE) {
      throw std::out_of_range("argument numeric value out of range");
    }
    return ret;
  }


  template <> inline
  double arg(const char* arg)
  {
    char* endptr = nullptr;
    errno = 0;
    double ret = std::strtod(arg, &endptr);
    if (endptr == arg) {
      std::ostringstream msg;
      msg << "unable to convert argument to integer: \"" << arg << "\"";
      throw std::invalid_argument(msg.str());
    }
    if (errno == ERANGE) {
      throw std::out_of_range("argument numeric value out of range");
    }
    return ret;
  }


  template <> inline
  const char* arg(const char* arg)
  {
    return arg;
  }


  template <> inline
  std::string arg(const char* arg)
  {
    return std::string(arg);
  }

}


inline
fmt_ostream::fmt_ostream(std::ostream& output)
: std::ostringstream(), output(output)
{
}


inline
fmt_ostream::~fmt_ostream()
{
  output << fmt_string(this->str());
}


#ifdef __unix__


inline
std::string fmt_string(const std::string& s)
{
  constexpr int read_end = 0;
  constexpr int write_end = 1;

  // TODO (vnguyen): This function overall needs to handle possible error
  // returns from the various syscalls.

  int read_pipe[2];
  int write_pipe[2];
  if (pipe(read_pipe) == -1) {
    return s;
  }
  if (pipe(write_pipe) == -1) {
    return s;
  }

  auto parent_pid = fork();
  bool is_fmt_proc = (parent_pid == 0);
  if (is_fmt_proc) {
    dup2(write_pipe[read_end], STDIN_FILENO);
    dup2(read_pipe[write_end], STDOUT_FILENO);
    close(write_pipe[read_end]);
    close(write_pipe[write_end]);
    close(read_pipe[read_end]);
    close(read_pipe[write_end]);
    const char* argv[] = {"fmt", NULL};
    execvp(const_cast<char*>(argv[0]), const_cast<char**>(argv));
  }

  close(write_pipe[read_end]);
  close(read_pipe[write_end]);
  auto fmt_write_fd = write_pipe[write_end];
  auto write_result = write(fmt_write_fd, s.c_str(), s.length());
  if (write_result != static_cast<ssize_t>(s.length())) {
    return s;
  }
  close(fmt_write_fd);

  auto fmt_read_fd = read_pipe[read_end];
  std::ostringstream os;
  char buf[64];
  while (true) {
    auto read_count = read(
      fmt_read_fd, reinterpret_cast<void*>(buf), sizeof(buf));
    if (read_count <= 0) {
      break;
    }
    os.write(buf, static_cast<std::streamsize>(read_count));
  }
  close(fmt_read_fd);

  return os.str();
}


#else // #ifdef __unix__


inline
std::string fmt_string(const std::string& s)
{
  return s;
}


#endif // #ifdef __unix__


} // namespace argagg


inline
std::ostream& operator << (std::ostream& os, const argagg::parser& x)
{
  for (auto& definition : x.definitions) {
    os << "    ";
    for (auto& flag : definition.flags) {
      os << flag;
      if (flag != definition.flags.back()) {
        os << ", ";
      }
    }
    os << std::endl;
    os << "        " << definition.help << std::endl;
  }
  return os;
}


#endif // ARGAGG_ARGAGG_ARGAGG_HPP
