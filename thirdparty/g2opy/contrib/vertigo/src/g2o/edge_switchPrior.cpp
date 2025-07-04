#include "edge_switchPrior.h"
using namespace std;

    bool EdgeSwitchPrior::read(std::istream &is)
    {
      is >> measurement();
      is >> information()(0,0);
      return true;
    }

    bool EdgeSwitchPrior::write(std::ostream &os) const
    {
      os << measurement() << " " << information()(0,0);
      return true;
    }

    void EdgeSwitchPrior::linearizeOplus()
    {
      _jacobianOplusXi[0]=-1.0;
    }

    void EdgeSwitchPrior::computeError()
    {
      const VertexSwitchLinear* s = static_cast<const VertexSwitchLinear*>(_vertices[0]);

      _error[0] = measurement() - s->x();

    }

