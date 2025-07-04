/*
 * switchVariableSigmoid.h
 *
 *  Created on: 08.08.2012
 *      Author: niko
 */

#ifndef SWITCHVARIABLESIGMOID_H_
#define SWITCHVARIABLESIGMOID_H_

#pragma once

#include <gtsam/base/DerivedValue.h>
#include <gtsam/base/Lie.h>

namespace vertigo {

  /**
   * SwitchVariableSigmoid is a wrapper around double to allow it to be a Lie type
   */
  struct SwitchVariableSigmoid : public DerivedValue<SwitchVariableSigmoid> {

    /** default constructor */
    SwitchVariableSigmoid() : d_(10.0) {};

    /** wrap a double */
    SwitchVariableSigmoid(double d) : d_(d) {
      if (d_ < -10.0) d_=-10.0;
      else if(d_>10.0) d_=10.0;
    };

    /** access the underlying value */
    double value() const { return d_; }

    /** print @param s optional string naming the object */
    inline void print(const std::string& name="") const {
      std::cout << name << ": " << d_ << std::endl;
    }

    /** equality up to tolerance */
    inline bool equals(const SwitchVariableSigmoid& expected, double tol=1e-5) const {
      return fabs(expected.d_ - d_) <= tol;
    }

    // Manifold requirements

    /** Returns dimensionality of the tangent space */
    inline size_t dim() const { return 1; }
    inline static size_t Dim() { return 1; }

    /** Update the SwitchVariableSigmoid with a tangent space update */
    inline SwitchVariableSigmoid retract(const Vector& v) const {
      double x = value() + v(0);

      if (x>10.0) x=10.0;
      else if (x<-10.0) x=-10.0;

      return SwitchVariableSigmoid(x);
    }

    /** @return the local coordinates of another object */
    inline Vector localCoordinates(const SwitchVariableSigmoid& t2) const { return Vector_(1,(t2.value() - value())); }

    // Group requirements

    /** identity */
    inline static SwitchVariableSigmoid identity() {
      return SwitchVariableSigmoid();
    }

    /** compose with another object */
    inline SwitchVariableSigmoid compose(const SwitchVariableSigmoid& p) const {
      return SwitchVariableSigmoid(d_ + p.d_);
    }

    /** between operation */
    inline SwitchVariableSigmoid between(const SwitchVariableSigmoid& l2,
        boost::optional<Matrix&> H1=boost::none,
        boost::optional<Matrix&> H2=boost::none) const {
      if(H1) *H1 = -eye(1);
      if(H2) *H2 = eye(1);
      return SwitchVariableSigmoid(l2.value() - value());
    }

    /** invert the object and yield a new one */
    inline SwitchVariableSigmoid inverse() const {
      return SwitchVariableSigmoid(-1.0 * value());
    }

    // Lie functions

    /** Expmap around identity */
    static inline SwitchVariableSigmoid Expmap(const Vector& v) { return SwitchVariableSigmoid(v(0)); }

    /** Logmap around identity - just returns with default cast back */
    static inline Vector Logmap(const SwitchVariableSigmoid& p) { return Vector_(1,p.value()); }

  private:
      double d_;
  };
}

#endif /* SWITCHVARIABLESIGMOID_H_ */
