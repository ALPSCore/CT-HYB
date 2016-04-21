#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <cassert>
#include <iostream>

//Implementation of Hoshen-Kopelman Algorithm
class Clustering {
public:
  Clustering(int N);
  void connect_vertices(int vertex1, int vertex2);
  void finalize_labeling();
  const std::vector<std::vector<int> >& get_cluster_members() const {
    assert(done_labeling);
    return cluster_members;
  }
  int get_num_clusters() const {
    assert(done_labeling);
    return cluster_members.size();
  }
  int get_cluster_label(int vertex) {
    assert(done_labeling);
    assert(vertex>=0&&vertex<N_);
    return cluster_labels_[vertex];
  }
  const std::vector<int>& get_cluster_labels() const {
    assert(done_labeling);
    return cluster_labels_;
  }

private:
  bool done_labeling;
  int get_true_label(int cluster_label);

  int N_;//number of vertices
  std::vector<int> cluster_belonging_to_, cluster_alias_, cluster_labels_;
  std::vector<std::vector<int> > cluster_members;

  //temp work space
  std::vector<int> visited_labels;
};

#endif
