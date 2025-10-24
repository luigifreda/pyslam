/*
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <memory>

namespace pyslam {

// Common base class for having shared_from_this() method. One should always virtually inherit from
// it.
class MultipleInheritableEnableSharedFromThis
    : public std::enable_shared_from_this<MultipleInheritableEnableSharedFromThis> {
  public:
    virtual ~MultipleInheritableEnableSharedFromThis() {}
};

template <class T>
class inheritable_enable_shared_from_this : virtual public MultipleInheritableEnableSharedFromThis {
  public:
    std::shared_ptr<T> shared_from_this() {
        return std::dynamic_pointer_cast<T>(
            MultipleInheritableEnableSharedFromThis::shared_from_this());
    }
    /* Utility method to easily downcast.
     * Useful when a child doesn't inherit directly from enable_shared_from_this
     * but wants to use the feature.
     */
    template <class Down> std::shared_ptr<Down> downcasted_shared_from_this() {
        return std::dynamic_pointer_cast<Down>(
            MultipleInheritableEnableSharedFromThis::shared_from_this());
    }
};

/*
 Usage example:

 class A: public inheritable_enable_shared_from_this<A>
 {
 public:
     void foo1()
     {
         auto ptr = shared_from_this();
     }
 };

 class B: public inheritable_enable_shared_from_this<B>
 {
 public:
     void foo2()
     {
         auto ptr = shared_from_this();
     }
 };

 class C: public inheritable_enable_shared_from_this<C>
 {
 public:
     void foo3()
     {
         auto ptr = shared_from_this();
     }
 };

 class D: public A, public B, public C
 {
 public:
     void foo()
     {
         auto ptr = A::downcasted_shared_from_this<D>();
     }
 };

*/

} // namespace pyslam