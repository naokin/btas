#ifndef _BTAS_QSXMERGE_INFO_H
#define _BTAS_QSXMERGE_INFO_H 1

#include <btas/btas_defs.h>
#include <btas/Quantum.h>

namespace btas
{

template<int N>
class QSXmergeInfo
{
public:
  typedef typename std::multimap<int, int>::const_iterator const_iterator;
  typedef typename std::pair<const_iterator, const_iterator> const_range;
private:
// boost
public:
  QSXmergeInfo()
  {
  }
  QSXmergeInfo(const TinyVector<Qshapes, N>& qshape, const TinyVector<Dshapes, N>& dshape)
  {
    reset(qshape, dshape);
  }

  void reset(const TinyVector<Qshapes, N>& qshape, const TinyVector<Dshapes, N>& dshape)
  {
    for(int i = 0; i < N; ++i)
      if(qshape[i].size() != dshape[i].size())
        BTAS_THROW(false, "btas::QSXmergeInfo::reset qshape and dshape have different block size");
    // packing qshape and dshape into 1d-array
    Qshapes qshape_pkd(1, Quantum::zero());
    Dshapes dshape_pkd(1, 1);
    for(int i = 0; i < N; ++i) {
      qshape_pkd = qshape_pkd * qshape[i];
      dshape_pkd = dshape_pkd * dshape[i];
    }
    // mapping packed index to merged quantum #
    std::map<Quantum, int> q_index_map;
    int n = 0;
    for(int i = 0; i < qshape_pkd.size(); ++i) {
      if(q_index_map.find(qshape_pkd[i]) == q_index_map.end()) {
        q_index_map.insert(std::make_pair(qshape_pkd[i], n++));
      }
    }
    // copying merged quantum #
    Qshapes qshape_mgd(n, Quantum::zero());
    for(typename std::map<Quantum, int>::iterator iqmap = q_index_map.begin(); iqmap != q_index_map.end(); ++iqmap) {
      qshape_mgd[iqmap->second] = iqmap->first;
    }
    // mapping packed index to merged index and computing merged block size
    m_index_map.clear();
    Dshapes dshape_mgd(n, 0);
    for(int i = 0; i < qshape_pkd.size(); ++i) {
      int j = q_index_map.find(qshape_pkd[i])->second;
      m_index_map.insert(std::make_pair(j, i));
      dshape_mgd[j] += dshape_pkd[i];
    }
    // save as members
    m_dshape = dshape;
    m_dshape_packed = dshape_pkd;
    m_dshape_merged = dshape_mgd;

    m_qshape = qshape;
    m_qshape_merged = qshape_mgd;
  }

  inline const TinyVector<Dshapes, N>& dshape() const { return m_dshape; }
  inline const Dshapes& dshape(int i) const { return m_dshape[i]; }

  inline const Dshapes& dshape_packed() const { return m_dshape_packed; }
  inline const int& dshape_packed(int i) const { return m_dshape_packed[i]; }

  inline const Dshapes& dshape_merged() const { return m_dshape_merged; }
  inline const int& dshape_merged(int i) const { return m_dshape_merged[i]; }

  inline const TinyVector<Qshapes, N>& qshape() const { return m_qshape; }
  inline const Qshapes& qshape(int i) const { return m_qshape[i]; }

  inline const Qshapes& qshape_merged() const { return m_qshape_merged; }
  inline const Quantum& qshape_merged(int i) const { return m_qshape_merged[i]; }

  inline const_iterator begin() const { return m_index_map.begin(); }
  inline const_iterator end() const { return m_index_map.end(); }
  inline const_iterator find(int i) const { return m_index_map.find(i); }

  inline const_range equal_range(int i) const { return m_index_map.equal_range(i); }

private:
  TinyVector<Dshapes, N>
    m_dshape;
  Dshapes
    m_dshape_packed;
  Dshapes
    m_dshape_merged;

  TinyVector<Qshapes, N>
    m_qshape;
  Qshapes
    m_qshape_merged;

  std::multimap<int, int>
    m_index_map;
};

};

#endif // _BTAS_QSXMERGE_INFO_H
