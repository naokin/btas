#ifndef __MPSXX_DMRG_SPIN_ADAPTED_SPIN_H
#define __MPSXX_DMRG_SPIN_ADAPTED_SPIN_H

namespace mpsxx {
namespace SpinAdapted {

class Spin {

private:

  struct _Spin_component {

    // 32-bits : 0000 0000 0000 0000 0000 0000 0000 0000
    //           -
    //           red. flag
    //                               ---------
    //                               S value   ---------
    //                                         Mz value
    static const unsigned int _MASK_RED_ = 0x80000000;
    static const unsigned int _MASK_S_   = 0x0000ff00;
    static const unsigned int _MASK_Sz_  = 0x000000ff;
    static const unsigned int _SHIFT_    = 8u;

    _Spin_component (unsigned int r = 0u) : rep_(r) { }

    bool reduced () const { return rep_ & _MASK_RED_; }

    unsigned int rep_;

  }; // struct _Spin_component

public:

  typedef _Spin_component value_type;

  static Spin zero ()
  {
  }

private:

  std::vector<_Spin_component> component_;

}; // struct Spin

} // namespace SpinAdapted
} // namespace mpsxx

#endif // __MPSXX_DMRG_SPIN_ADAPTED_SPIN_H
