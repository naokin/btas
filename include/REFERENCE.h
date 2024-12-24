// ---------------------------------------------------------------------------------------------------- 
// Class/Function REFERENCE
// ---------------------------------------------------------------------------------------------------- 

// tensor-view object and tensor-iterator


// quantum-number/symmetry object
{
  // static function to provide "zero" quantum-number
  qnum_type q0 = Q::zero();

  // multiplying operators
  qnum_type q1(...);
  qnum_type q2(...);
  qnum_type qm = q1*q2; // or, qnum_type qm = q1+q2;

  // conjugation
  qnum_type qc = qm.conj(); // or, qnum_type qc = -qm;

  // rational operators
  if(q1 == q2) ...;
  if(q1 < qm) ...;

  // 
}


// sparse tensor object
