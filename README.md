# TIOC-TWD
A tree-based incremental overlapping clustering method using the three-way decision theory



# 核心算法

**生成聚类：**

void SOC_TWD(vector<vector<double>> *U, int N, int attrs, double alpha, double beta, double threshold,
	vector<RP*> *R, Graph **Graph2, vector<Cluster*> **clusters2);

**创建查找树：**

void CST(vector<RP*> *R, int attrs, Node *Root, double threshold = 0.9);

**找到邻接点：**

void FindingNeighbors(Node* Root, RP* r_wait, double threshold, 
	int attrs, vector<RP*>* ret, vector<Node*> *Path);

**更新聚类：**

void UpdatingClustering(vector<RP*> *R, double alpha, double beta, double threshold,
	int attrs, Node* root, vector<vector<double>> *U2, Graph *G, vector<Cluster*> *clusters);




