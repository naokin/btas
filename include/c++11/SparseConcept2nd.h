


template<typename T>
class SpTensor {

private:

  SparseShape shape_;

  std::vector<Tile> store_;

};



template<class Q>
class GS_qnum_ {

private:

  static std::vector<Q> qnum_list_;

  static btas::Tensor<unsigned int> table_;

public:

  typedef unsigned int qnum_index_type;

private:

  qnum_index_type index_;

};

