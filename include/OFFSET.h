
{
  ordinal_type n = tn_stride_.size();
  difference_type m = tn_stride_.ordinal(idx)+offs;
  if(m < 0) { m = 0; }
  if(m > n) { m = n; }
  index_type index_to_ = tn_stride_.index(m);
  //
  for(size_t i = 0; i < N; ++i) current_ += (index_to_[i]-index_[i])*stride_hack_[i];
}


Iterator offset (index_type& idx, difference_type n)
{
  const extent_type& ext; // extent of tensor view
  const stride_type& str; // stride hack
  const size_t N = ext.size();

  difference_type p = 0;
  size_t m;

  for(size_t i = N-1; n > 0 && i > 0; --i) {
    n += idx[i];
    m = n % ext[i];
    n /= ext[i];
    p = (m-idx[i])*str[i];
    idx[i] = m;
  }
  idx[0] += n;
  //
  if(idx[0] >= ext[0]) {
    // make iterator to the end
    for(size_t i = N-1; i > 0; --i) {
      p -= idx[i]*str[i];
      idx[i] = 0;
    }
    idx[0] = ext[0];
  }
}

{
  // where n < 0
  difference_type p = 0;
  size_t m;

  for(size_t i = N-1; n < 0 && i > 0; --i) {
    n += idx[i];
    m = n % ext[i];
    n /= ext[i];
    if(m < 0) { m += ext[i]; --n; }
    p = (m-idx[i])*str[i];
    idx[i] = m;
  }
  //
  if(idx[0] 
  idx[0] += n;

  if(idx[0] < 0) {
    // make iterator to the end
    for(size_t i = N-1; i > 0; --i) {
      p -= idx[i]*str[i];
      idx[i] = 0;
    }
    idx[0] = ext[0];
  }
}
