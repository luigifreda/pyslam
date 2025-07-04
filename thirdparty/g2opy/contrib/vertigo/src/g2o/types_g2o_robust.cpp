#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"

#include "edge_switchPrior.h"
#include "edge_se2Switchable.h"
#include "edge_se2MaxMixture.h"
#include "edge_se3Switchable.h"
#include "vertex_switchLinear.h"


using namespace g2o;


void G2O_ATTRIBUTE_CONSTRUCTOR init_types_veloc(void)
{
  Factory* factory = Factory::instance();
  factory->registerType("EDGE_SWITCH_PRIOR", new HyperGraphElementCreator<EdgeSwitchPrior>);
  factory->registerType("EDGE_SE2_SWITCHABLE", new HyperGraphElementCreator<EdgeSE2Switchable>);
  factory->registerType("EDGE_SE2_MAXMIX", new HyperGraphElementCreator<EdgeSE2MaxMixture>);
  factory->registerType("EDGE_SE3_SWITCHABLE", new HyperGraphElementCreator<EdgeSE3Switchable>);
  factory->registerType("VERTEX_SWITCH", new HyperGraphElementCreator<VertexSwitchLinear>);



  g2o::HyperGraphActionLibrary* actionLib = g2o::HyperGraphActionLibrary::instance();

#ifdef G2O_HAVE_OPENGL
      actionLib->registerAction(new EdgeSE2SwitchableDrawAction);
      actionLib->registerAction(new EdgeSE2MaxMixtureDrawAction);
      actionLib->registerAction(new EdgeSE3SwitchableDrawAction);
#endif


}
