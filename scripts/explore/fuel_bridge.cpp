// Adapter for the UNMODIFIED FUEL Polynomial class (FUEL GPL-3.0).
// Rest-to-rest quintic boundary coefficients; no ROS or hidden map input.
#include <poly_traj/polynomial_traj.h>
#include <algorithm>
extern "C" int fuel_eval(const double* a,const double* b,double duration,double t,int derivative,double* out) {
  if(duration<=0 || derivative<0 || derivative>3) return 1;
  fast_planner::Polynomial::Vector6d coefficients[3];
  for(int d=0;d<3;d++) {
    coefficients[d].setZero(); coefficients[d][0]=a[d];double delta=b[d]-a[d];
    coefficients[d][3]=10*delta/std::pow(duration,3);
    coefficients[d][4]=-15*delta/std::pow(duration,4);
    coefficients[d][5]=6*delta/std::pow(duration,5);
  }
  fast_planner::Polynomial p(coefficients[0],coefficients[1],coefficients[2],duration);
  auto v=p.evaluate(std::max(0.,std::min(t,duration)),derivative);
  for(int d=0;d<3;d++) out[d]=v[d];
  return 0;
}
