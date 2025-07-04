/*
 * vertex_switchLinear.h
 *
 *  Created on: 17.10.2011
 *      Author: niko
 */


#pragma once

#include "g2o/core/base_vertex.h"
#include <math.h>



    class VertexSwitchLinear : public g2o::BaseVertex<1, double>
    {

    public:
      VertexSwitchLinear() { setToOrigin(); };

      virtual void setToOrigin();

      virtual void oplus(double* update);

      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;
      virtual void setEstimate(double &et);


      double x() const { return _x; };


      //! The gradient at the current estimate is always 1;
      double gradient() const { return 1; } ;

    private:
      double _x;

    };
