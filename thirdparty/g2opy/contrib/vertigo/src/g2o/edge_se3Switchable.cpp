/*
 * edge_se3Switchable.cpp
 *
 *  Created on: 17.10.2011
 *      Author: niko
 */

#include "edge_se3Switchable.h"
#include "vertex_switchLinear.h"
#include <GL/gl.h>
#include "g2o/math_groups/se3quat.h"


using namespace std;
using namespace Eigen;


// ================================================
EdgeSE3Switchable::EdgeSE3Switchable() : g2o::BaseMultiEdge<6, g2o::SE3Quat>()
{
  resize(3);
  _jacobianOplus[0].resize(6,6); 
  _jacobianOplus[1].resize(6,6);
  _jacobianOplus[2].resize(6,1);

}
// ================================================
bool EdgeSE3Switchable::read(std::istream& is)
  {
    for (int i=0; i<7; i++)
    is >> measurement()[i];
    measurement().rotation().normalize();
    inverseMeasurement() = measurement().inverse();

    for (int i=0; i<6; i++)
      for (int j=i; j<6; j++) {
        is >> information()(i,j);
        if (i!=j)
          information()(j,i) = information()(i,j);
      }
    return true;

  }
// ================================================
bool EdgeSE3Switchable::write(std::ostream& os) const
{
    g2o::Vector7d p = measurement().toVector();
    os << p.x() << " " << p.y() << " " << p.z();
    for (int i = 0; i < 6; ++i)
      for (int j = i; j < 6; ++j)
        os << " " << information()(i, j);
    return os.good();
}


// forward declaration for the analytic jacobian
namespace g2o
{
  void  jacobian_3d_qman ( Matrix<double, 6, 6> &  Ji , Matrix<double, 6, 6> &  Jj, const double&  z11 , const double&  z12 , const double&  z13 , const double&  z14 , const double&  z21 , const double&  z22 , const double&  z23 , const double&  z24 , const double&  z31 , const double&  z32 , const double&  z33 , const double&  z34 , const double&  xab11 , const double&  xab12 , const double&  xab13 , const double&  xab14 , const double&  xab21 , const double&  xab22 , const double&  xab23 , const double&  xab24 , const double&  xab31 , const double&  xab32 , const double&  xab33 , const double&  xab34 );
}


// ================================================
void EdgeSE3Switchable::linearizeOplus()
{

    g2o::VertexSE3* from = static_cast<g2o::VertexSE3*>(_vertices[0]);
    g2o::VertexSE3* to = static_cast<g2o::VertexSE3*>(_vertices[1]);
    const VertexSwitchLinear* vSwitch = static_cast<const VertexSwitchLinear*>(_vertices[2]);

    Matrix3d izR        = inverseMeasurement().rotation().toRotationMatrix();
    const Vector3d& izt = inverseMeasurement().translation();

    g2o::SE3Quat iXiXj         = from->estimate().inverse() * to->estimate();
    Matrix3d iRiRj        = iXiXj.rotation().toRotationMatrix();
    const Vector3d& ititj = iXiXj.translation();

    Matrix<double, 6, 6> Ji, Jj;

    g2o::jacobian_3d_qman ( Ji, Jj,
              izR(0,0), izR(0,1), izR(0,2), izt(0),
              izR(1,0), izR(1,1), izR(1,2), izt(1),
              izR(2,0), izR(2,1), izR(2,2), izt(2),
              iRiRj(0,0), iRiRj(0,1), iRiRj(0,2), ititj(0),
              iRiRj(1,0), iRiRj(1,1), iRiRj(1,2), ititj(1),
              iRiRj(2,0), iRiRj(2,1), iRiRj(2,2), ititj(2));

    _jacobianOplus[0] = Ji;
    _jacobianOplus[1] = Jj;


    _jacobianOplus[0]*=vSwitch->estimate();
    _jacobianOplus[1]*=vSwitch->estimate();


    // derivative w.r.t switch vertex
    _jacobianOplus[2].setZero();

    g2o::SE3Quat delta = _inverseMeasurement * (from->estimate().inverse()*to->estimate());
    ErrorVector error;
    error.head<3>() = delta.translation();
    // The analytic Jacobians assume the error in this special form (w beeing positive)
    if (delta.rotation().w() < 0.)
      error.tail<3>() =  - delta.rotation().vec();
    else
      error.tail<3>() =  delta.rotation().vec();

    _jacobianOplus[2] = error * vSwitch->gradient();

}


// ================================================
void EdgeSE3Switchable::computeError()
{

    const VertexSwitchLinear* v3 = static_cast<const VertexSwitchLinear*>(_vertices[2]);


    const g2o::VertexSE3* v1 = dynamic_cast<const g2o::VertexSE3*>(_vertices[0]);
    const g2o::VertexSE3* v2 = dynamic_cast<const g2o::VertexSE3*>(_vertices[1]);
    g2o::SE3Quat delta = _inverseMeasurement * (v1->estimate().inverse()*v2->estimate());
    _error.head<3>() = delta.translation()* v3->estimate();
    // The analytic Jacobians assume the error in this special form (w beeing positive)
    if (delta.rotation().w() < 0.)
      _error.tail<3>() =  - delta.rotation().vec()* v3->estimate();
    else
      _error.tail<3>() =  delta.rotation().vec()* v3->estimate();;

}


#ifdef G2O_HAVE_OPENGL
  EdgeSE3SwitchableDrawAction::EdgeSE3SwitchableDrawAction(): DrawAction(typeid(EdgeSE3Switchable).name()){}

  g2o::HyperGraphElementAction* EdgeSE3SwitchableDrawAction::operator()(g2o::HyperGraph::HyperGraphElement* element,
               g2o::HyperGraphElementAction::Parameters* /*params_*/){
    if (typeid(*element).name()!=_typeName)
      return 0;
    EdgeSE3Switchable* e =  static_cast<EdgeSE3Switchable*>(element);


    g2o::VertexSE3* fromEdge = static_cast<g2o::VertexSE3*>(e->vertices()[0]);
    g2o::VertexSE3* toEdge   = static_cast<g2o::VertexSE3*>(e->vertices()[1]);
    VertexSwitchLinear* s   = static_cast<VertexSwitchLinear*>(e->vertices()[2]);

    glColor3f(s->estimate()*1.0,s->estimate()*0.1,s->estimate()*0.1);
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glVertex3f(fromEdge->estimate().translation().x(),fromEdge->estimate().translation().y(),fromEdge->estimate().translation().z());
    glVertex3f(toEdge->estimate().translation().x(),toEdge->estimate().translation().y(),toEdge->estimate().translation().z());
    glEnd();
    glPopAttrib();
    return this;
  }
#endif
