#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <cassert>
#include <iostream>

//Implementation of Hoshen-Kopelman Algorithm
class Clustering {
public:
  Clustering(int N) : done_labeling(false), N_(N), cluster_belonging_to_(N), cluster_alias_(N) {
    //In the begining, each vertex forms its own cluster.
    for (int vertex=0; vertex<N_; ++vertex) {
      cluster_belonging_to_[vertex] = vertex;
    }
    for (int cluster=0; cluster<N_; ++cluster) {
      cluster_alias_[cluster] = cluster;
    }

    visited_labels.reserve(N);
  }

  //If two vertices belong to different cluster, merge them into a single cluster
  void connect_vertices(int vertex1, int vertex2) {
    assert(!done_labeling);
    //first, we see if vertex1 and vertex2 belong to the same cluster
    const int cluster1 = get_true_label(cluster_alias_[vertex1]);
    const int cluster2 = get_true_label(cluster_alias_[vertex2]);
    if (cluster1==cluster2) {
      return;
    } else if (cluster1<cluster2) {
      cluster_alias_[cluster2] = cluster1;
    } else {
      cluster_alias_[cluster1] = cluster2;
    }
  }

  //Assign true labels to all clusters, and find members of each cluster.
  void finalize_labeling() {
    assert(!done_labeling);
    std::vector<int> valid_labels;
    std::vector<bool> flag(N_,false);
    for (int i=0; i<N_; ++i) {
      cluster_alias_[i] = get_true_label(i);
      if (!flag[cluster_alias_[i]]) {
        flag[cluster_alias_[i]] = true;
        valid_labels.push_back(cluster_alias_[i]);
      }
    }
    const int num_valid_labels = valid_labels.size();

    cluster_members.resize(num_valid_labels);
    cluster_labels_.resize(N_);
    std::vector<int> label_map(N_);
    for (int label=0; label<num_valid_labels; ++label) {
      label_map[valid_labels[label]] = label;
    }
    for (int i=0; i<N_; ++i) {
      const int new_label = cluster_labels_[i] = label_map[cluster_alias_[i]];
      assert(new_label>=0);
      assert(new_label<cluster_members.size());
      cluster_members[new_label].push_back(i);
    }

    done_labeling = true;
  }

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

  //Get the true label of a given cluster
  int get_true_label(int c_label) {
    assert(!done_labeling);
    assert(c_label<N_);
    if (c_label==cluster_alias_[c_label]) {
      return c_label;
    }

    visited_labels.resize(0);
    int c_label_tmp = c_label;
    while (c_label_tmp != cluster_alias_[c_label_tmp]) {
      visited_labels.push_back(c_label_tmp);
      c_label_tmp = cluster_alias_[c_label_tmp];
    }
    for (int hist=0; hist<visited_labels.size(); ++hist) {
      cluster_alias_[visited_labels[hist]] = c_label_tmp;
    }
    return c_label_tmp;
  }

  int N_;//number of vertices
  std::vector<int> cluster_belonging_to_, cluster_alias_, cluster_labels_;
  std::vector<std::vector<int> > cluster_members;

  //temp work space
  std::vector<int> visited_labels;
};

#endif
