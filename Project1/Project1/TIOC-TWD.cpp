#include"initStructure.h"
using namespace std;


//####################################################################################
//	算法1 2 的公共方法实现
//####################################################################################

//邻居集的邻居数目降序排序
bool cmpNeighbor(const Neighbor *nFirst, const Neighbor *nSecond)
{
	if (nFirst->count > nSecond->count) //由大到小排序 注意 == 时候  必须返回false
	{
		return true;
	}
	return false;
}

//获取根据记录号获取 记录的邻居集合
void getNeighborByRecord(vector<Neighbor*> &neighbors, int record, vector<int>& nei)
{
	int flag = 0, i = 0;
	for (; i < neighbors.size(); i++)
	{
		if (neighbors[i]->center == record)
		{
			flag = 1;
			nei = neighbors[i]->Nei;
			break;
		}

	}
}

/*
-- 计算代表点之间的欧里几德距离
*/
double getRPDistance(RP *x, RP *y, int attrs)
{
	double ret, sum, temp;
	ret = sum = temp = 0;
	for (int i = 0; i < attrs; i++)
	{
		double xi = x->getLeft(i);
		double yi = y->getLeft(i);
		if (xi == MAX || yi == MAX)
		{
			return MAX;
		}
		temp = xi - yi;
		temp = pow(temp, 2);
		sum += temp;
	}
	ret = sqrt(sum);
	sum = 0;
	for (int j = 0; j < attrs; j++)
	{
		temp = x->getRight(j) - y->getRight(j);
		temp = pow(temp, 2);
		sum += temp;
	}
	ret += sqrt(sum);
	return ret;
}

/*
计算两个代表点的相似度
Similarity = 【  Cover(i) 交 Cover(j) 】 / min{Cover(i) ，Cover(j)}
*/
double computeSimilarity(RP* ri, RP* rj)
{
	set<int> *ri_cover = new set<int>();
	set<int> *rj_cover = new set<int>();
	vector<int>::iterator insert_iterator;
	set<int> *res = new set<int>();//存储交集结果

	double min, inter, isize, jsize;
	double ret;
	for (int i = 0; i < ri->getCoverSize(); i++)
	{
		ri_cover->insert(ri->getCoverVal(i));
	}
	for (int j = 0; j < rj->getCoverSize(); j++)
	{
		rj_cover->insert(rj->getCoverVal(j));
	}
	jsize = rj_cover->size();
	isize = ri_cover->size();
	min = isize < jsize ? isize : jsize;
	//
	//std::set_intersection(ri_cover->begin(), ri_cover->end(), ri_cover->begin(), ri_cover->end(), res->begin());
	inter = res->size();
	ret = inter / min;
	return ret;
}

//####################################################################################
//算法3  4 公共算法的实现
//####################################################################################
//获取结点数目
int getINodesI(set<Node*> *Nodes)
{
	return Nodes->size();
}
//RP结点的比较函数
bool cmpRP(RP *r1, RP *r2)
{
	if (r1->getLeft(1) > r2->getLeft(1))
		return true;
	return false;
}

/*
function：判断一个代表点的第layer个属性  是否与Node中的代表点相似
	rp：代表点
	node：结点
	layer：第几层 （根结点为第0层）
*/
bool isSimilarity(Node * node, RP* rp, int layer)
{
	double rp_left = rp->getLeft(layer);
	double rp_right = rp->getRight(layer);
	vector<RP*> *R = node->R;
	vector<RP*>::iterator it;
	for (it = R->begin(); it != R->end(); it++)
	{
		/*由于
		rp 第layer个属性的下界值  大于  node中所有代表点的第layer个属性的下界值
		（因为排过序了）
		所以判断  （A）rp第layer个属性的下界值  和  （B）node中所有代表点的第layer个属性的上界值 就好
		如果（B）>（A）  说明有相似返回 true
		*/
		if (((*it)->right)[layer] > rp_left)
			return true;
	}
	return false;
}

/*
	将 newNode结点 融合到 treeNode结点
	俩个结点融合后
	：父节点不变,但是父亲结点对应孩子结点newNode需要删除
	：代表点集合融合
	：孩子结点融合
	：上下界重新计算
*/
void merge(Node* treeNode, Node* newNode)
{
	//重新计算上下界
	if (treeNode->low > newNode->low)
		treeNode->low = newNode->low;
	if (treeNode->high < newNode->high)
		treeNode->high = newNode->high;
	vector<RP*> *R = newNode->R;
	vector<RP*>::iterator it_r;
	//代表点的融合
	for (it_r = R->begin(); it_r != R->end(); it_r++)
		treeNode->add_RP(*it_r);

	//孩子结点的融合
	vector<Node*> *sons_Node = newNode->sons_Node;
	vector<Node*>::iterator it_s;
	for (it_s = sons_Node->begin(); it_s != sons_Node->end(); it_s++)
		treeNode->add_SonNode(*it_s);
	//删除father结点的孩子结点newNode
	Node* father = treeNode->father_Node;
	vector<Node*> *f_s = father->sons_Node;
	vector<Node*>::iterator it_son;
	bool flag = false;
	for (it_son = f_s->begin(); it_son != f_s->end(); it_son++)
	{
		if (*it_son == newNode)
		{
			f_s->erase(it_son);
			flag = true;
			break;
		}
	}
}

//联合
bool isSimilarityNN(Node * node, Node * nnode, int layer)
{
	RP *rp = nnode->R->at(0);
	return isSimilarity(node, rp, layer);
}

//结点的比较函数
bool cmpNN(Node *r1, Node *r2)
{
	if (r1->low > r2->low)
		return true;
	return false;
}

/*
Input:
	R:	代表点集合
	G:
	alpha:
	beta:
	threshold: 阈值
Output:
	C:	聚类
*/
double getDistanceRX(int rd, RP* rp, double **U2, int attrs)
{
	double ret, sum, temp;
	int j = rp->representationPoint;
	X x = X(*(U2 + rd - 1), attrs);
	X y = X(*(U2 + j - 1), attrs);
	ret = sum = temp = 0;
	for (int i = 0; i < attrs; i++)
	{
		temp = x.getVal(i) - y.getVal(i);
		temp = pow(temp, 2);
		sum += temp;
	}
	ret = sqrt(sum);
	return ret;
}

//####################################################################################
//  打印方法（用于数据显示）
//####################################################################################
//打印邻居集合
void printNei(const vector<Neighbor*> *neis)
{
	cout << "==========================================" << endl;
	cout << "打印邻居集合。。。。。。。" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < neis->size(); i++)
	{
		Neighbor* nei = (*neis)[i];
		cout << "中心点：" << nei->center << " ; ";
		cout << '\t' << "邻居数目：" << nei->count << " ; ";
		for (int j = 0; j < nei->Nei.size(); j++)
		{
			cout << (nei->Nei)[j] << " , ";
		}
		cout << endl;
	}
}

//打印代表点集合
void printR(const vector<RP*> *R, int attrs)
{
	cout << "==========================================" << endl;
	cout << "打印代表点集合。。。。。。。" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < R->size(); i++)
	{
		RP* rp = (*R)[i];
		cout << "代表点：" << rp->representationPoint << endl;
		cout << '\t' << "覆盖域：";
		for (int j = 0; j < rp->Cover.size(); j++)
		{
			cout << (rp->Cover)[j] << " , ";
		}
		cout << endl << '\t' << "属性下界：";
		for (int j = 0; j < attrs; j++)
		{
			cout << (rp->left)[j] << '\t';
		}
		cout << endl << '\t' << "属性上界：";
		for (int j = 0; j < attrs; j++)
		{
			cout << (rp->right)[j] << '\t';
		}
		cout << endl;
	}
}

//打印记录状态矩阵
void printState(Distance *distance)
{
	cout << "==========================================" << endl;
	cout << "打印记录状态矩阵:" << endl;
	cout << "==========================================" << endl;
	for (int i = 1; i < distance->NUM + 1; i++)
		cout << distance->state[i] << "  ";
	cout << endl;
}
//打印代表点de 图
void printG(Graph *graph)
{
	cout << "==========================================" << endl;
	cout << "打印代表点所组成的 Strong AND Weak图:" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < graph->RN; i++)
	{
		for (int j = 0; j < graph->RN; j++)
		{
			cout << *(*((graph->G) + i) + j) << "  ";
		}
		cout << endl;
	}
}

//打印强关联子图 和 弱关联子图
void printGR(Graph *graph)
{
	cout << "==========================================" << endl;
	cout << "打印强关联子图 和 弱关联子图" << endl;
	cout << "==========================================" << endl;
	// vector<int> 中存的是在矩阵行或列中的位置   可以   映射到代表点的指针
	list<vector<int>*>::iterator it;
	//
	list<set<int>*>::iterator itW;

	set<int>::iterator itW2;
	int itr = 0;
	for (it = graph->subGraphs.begin(), itW = graph->subGraphsWeak.begin();
		it != graph->subGraphs.end();
		it++, itW++)
	{
		itr++;
		cout << "第" << itr << "个强关联子图：" << '\t';
		for (int j = 0; j < (*it)->size(); j++)
		{
			cout << graph->getMapping((*(*it))[j])->representationPoint << "  ";
		}

		cout << '\t' << "第" << itr << "个弱关联子图：" << '\t';
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//根据 存储的矩阵行或列中的位置      映射到    代表点的指针
			cout << graph->getMapping(*itW2)->representationPoint << "  ";
		}
		cout << endl;
	}
}

void printCluster(vector<Cluster*> *clusters)
{
	cout << "==========================================" << endl;
	cout << "打印聚类" << endl;
	cout << "==========================================" << endl;
	vector<Cluster*>::iterator it;
	int itr = 0;
	for (it = clusters->begin(); it != clusters->end(); it++)
	{
		itr++;
		cout << "第" << itr << "个聚类：" << '\t';
		Cluster* cl = *it;
		set<int>::iterator itPos;
		for (itPos = cl->POS->begin(); itPos != cl->POS->end(); itPos++)
		{
			cout << *itPos << '\t';
		}
		cout << endl;
	}
}


//####################################################################################
//  四大算法
//####################################################################################
/*
算法1：
	U 记录矩阵
	N 记录数目
	attrs 属性数目
	alpha
	beta
	threshold 阈值
*/
void SOC_TWD(double *U[], int N, int attrs, double alpha, double beta, double threshold)
{
	//----------------------------------------------------------------------------
	//--初始化
	//----------------------------------------------------------------------------
	// R 存放代表点的容器
	vector<RP*> *R = new vector<RP*>();
	// neighbors 存放各个记录的所有邻居的容器
	vector<Neighbor*> neighbors;
	// distance 存放两个记录之间距离的矩阵
	Distance *distance = new Distance(N);

	//vector<RNeighbor*> rneighbors;
	vector<vector<RP*>*> rneighbors;

	/*----------------------------------------------------------------------------
	--计算距离矩阵 计算每个对象（记录）之间的距离
	----------------------------------------------------------------------------*/
	distance->setAllDistance(U, N, attrs);
	/*----------------------------------------------------------------------------
	--计算Xi的邻居
	--所有的记录的邻居都要计算
	----------------------------------------------------------------------------*/
	for (int i = 1; i < N + 1; i++)
	{

		Neighbor *nei = new Neighbor(i);
		for (int j = 1; j < N + 1; j++)
		{
			if (i == j)//邻居不需要计算自身
				;
			else
			{//判断两个对象的距离是否 <= threshold
				if (distance->getDisVal(i, j) <= threshold)
				{//小于等于则加入到该对象的邻居集中
					nei->addNeighbor(j);
				}
			}
		}
		//存储 Xi 的邻居数目
		nei->count = nei->Nei.size();
		// 存入所有点的邻居集
		neighbors.push_back(nei);
	}
	/*----------------------------------------------------------------------------
	-- 对neighbors排序
	-- 按每个Xi（对象|记录）的邻居数目降序排序
	----------------------------------------------------------------------------*/

	printNei(&neighbors);
	std::sort(neighbors.begin(), neighbors.end(), cmpNeighbor);
	printNei(&neighbors);
	/*----------------------------------------------------------------------------
	-- 创建 R 代表点集合,并将所有代表点加入进去
	-- RP 包括 代表点   覆盖区域   属性上 和 下界
	----------------------------------------------------------------------------*/
	//判断 距离矩阵 中是否还存在Row（以下while循环中会有假删除操作）
	while (distance->getRows() != 0)
	{
		//选择第一行（距离矩阵 未删除数据的第一行）
		int record = distance->getFirstRecord();
		if (record == -1)//说明已经删完了
			break;
		// 存放邻居集合
		vector<int> nei;
		// 获取nei  根据记录号 在 存放所有记录邻居集合 中获取对应邻居集合
		getNeighborByRecord(neighbors, record, nei);
		//生成代表点 设置代表记录  和  覆盖区域   初始化属性值上下界
		RP *rp = new RP(record, nei, attrs);
		for (int i = 0; i < rp->Cover.size(); i++)
		{
			int reco = (rp->Cover)[i];
			//设置代表点的 属性上|下界
			for (int attr = 0; attr < attrs; attr++)
			{
				//Cover中保存的是对象号  +1第一行存的是记录号
				// U 是从 0 开始计算   而对象号从 1 开始算 ，所以*(U + reco -1)
				double attrVal = *(*(U + reco - 1) + 1 + attr);
				if (rp->getLeft(attr) > attrVal)
					rp->setLeft(attr, attrVal);
				if (rp->getRight(attr) < attrVal)
					rp->setRight(attr, attrVal);;
			}
			distance->deleteRows(reco);
		}
		R->push_back(rp);
	}

	printState(distance);
	printR(R, attrs);
	/*----------------------------------------------------------------------------
	-- 创建G图   并且   计算相似度
	-- 使用代表点 当 做图中的结点
	//--------------------------------------------------------------------------*/
	Graph *G = new Graph(R);
	for (int i = 0; i < R->size(); i++)
	{
		RP* ri = (*R)[i];
		double similarityRiRj;//存储相似度
		//RNeighbor* rnei = ri->rpNeighbor;
		vector<RP*> *rnei = ri->rpNeighbor;
		for (int j = 0; j < R->size(); j++)
		{
			RP* rj = R->at(j);
			if (i == j)
				;
			else
			{
				//获得代表点之间的距离
				double distance = getRPDistance(ri, rj, attrs);
				if (distance <= 2 * threshold)
				{
					rnei->push_back(rj);
				}
				//计算出代表点之间的相似度
				similarityRiRj = computeSimilarity(ri, rj);
				if (similarityRiRj >= alpha)
				{
					//strong link
					G->addStrongLink(ri, rj);
				}
				if (similarityRiRj < alpha && similarityRiRj >= beta)
				{
					//weak link
					G->addWeakLink(ri, rj);
				}
			}
		}
		rneighbors.push_back(rnei);
	}
	printG(G);
	//----------------------------------------------------------------------------
	// 获取强连通子图
	// 并且生成子图
	//----------------------------------------------------------------------------
	G->BFS();
	printGR(G);
	//----------------------------------------------------------------------------
	//----------------------生成聚类集合------------------------------------------
	//----------------------------------------------------------------------------
	vector<Cluster*> *clusters = new vector<Cluster*>();
	list<vector<int>*>::iterator it; //声明一个迭代器 用于迭代Strong link的
	list<set<int>*>::iterator itW; //声明一个迭代器 用于迭代weak link的
	int newN = 1;
	for (it = G->subGraphs.begin(), itW = G->subGraphsWeak.begin();
		it != G->subGraphs.end();
		it++, itW++) {
		Cluster *Cx = new Cluster(newN);

		for (int i = 0; i < (*it)->size(); i++)
		{
			//根据 存储的矩阵行或列中的位置      映射到    代表点的指针
			RP* rp = G->getMapping((*(*it))[i]);//(*(*it))[i]    (*it)->at(i)
			Cx->addCoverToPOS(rp);
		}
		set<int>::iterator itW2;
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//根据 存储的矩阵行或列中的位置      映射到    代表点的指针
			RP* rp = G->getMapping(*itW2);
			Cx->addCoverToBND(rp);
		}
		newN++;
		clusters->push_back(Cx);
	}
	printCluster(clusters);
}


/*
算法2：
Creating the searching tree
	:使用层序的方式创建searching tree
input ：
	R：代表点集合
	A：属性
	threshold：阈值
output：
	root：root of researching tree
*/
void CST(vector<RP*> *R, int *A,int attrs, double threshold)
{
	/*
		vector<set<Node*>>					存放所有层的 结点集合
		set<Node*>			Node(i)			某一层的结点集合
		Node				node			一个结点
	*/
	vector<set<Node*>> *LayerNodes = new vector<set<Node*>>();
	int layer = 0;
	// 根结点的父结点为NULL
	Node *Root = new Node(layer,NULL);
	Root->R = R;
	double lastLayer, curLayer;

	// 创建一个新的层  第0层
	set<Node*> Node0;
	Node0.insert(Root);
	LayerNodes->push_back(Node0);

	lastLayer = 0.1;
	curLayer = getINodesI( &Node0 );

	Node *tempFather;
	while (lastLayer/curLayer < threshold || layer < attrs)
	{
		//=========================================================================
		// 遍历第 i 层中的所有结点(一个结点 中 包含多个代表点)
		//=========================================================================
		set<Node*>::iterator it_node_ij;
		for ( it_node_ij = LayerNodes->at(layer).begin();
			it_node_ij != LayerNodes->at(layer).end();
			it_node_ij++ )
		{	//当前结点（第 i 层）设置为父亲结点
			tempFather = (*it_node_ij);
			//=========================================================================
			//	根据第i+1个属性对 第i层的结点中的代表点 进行升序排序
			//	R中的代表点  按照下界值的增序排序   所以后面的结点下界  >  前面结点下界
			//	cmpRP函数  根据第 layer 层（属性）进行排序
			//=========================================================================
			vector<RP*> *R = (*it_node_ij)->R;
			// 对上界还是下界排序 ？ 第几个属性？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？  
			std::sort(R->begin(),R->end(), cmpRP);
			//=========================================================================
			//	创建第 i+1 层新结点（设置它的层数 和 父亲结点）
			//	将上层父亲节点 经过排序后的代表点集合的 第一个 代表点    加入到该节点中
			//=========================================================================
			Node *node_ij_1 = new Node(layer, tempFather);
			tempFather->add_SonNode( node_ij_1 );
			node_ij_1->add_RP( *(R->begin()) );
			//=========================================================================
			//	通过代表点的相似度对第 i+1 层结点进行划分
			//=========================================================================
			vector<RP*>::iterator it_rp;
			for (it_rp = R->begin() + 1; it_rp != R->end(); it_rp++)
			{
				//=========================================================================
				// 计算相似性   使用Definition 5
				//=========================================================================
				bool similarity = isSimilarity(node_ij_1, *it_rp,layer);
				//判断相似性
				if (similarity)
				{	//代表点和结点相似 则将代表点加入到结点包含的集合之中
					node_ij_1->add_RP(*it_rp);
				}
				else//代表点和结点不相似 则创建新的结点
				{
					//=========================================================================
					// 计算该结点（node_ij_1）的 的第layer属性的low 和 high值
					// 并插入到树的第 layer 层
					//=========================================================================
					node_ij_1->computeLowAndHigh(layer);
					// 插入到查找树的第layer层
					((*LayerNodes)[layer]).insert(node_ij_1);
					//=========================================================================
					// 创建一个新的结点 设置它的层数 和 父亲结点
					// 并设置其包含的初始代表点
					//=========================================================================
					Node *node_ij_2 = new Node(layer, tempFather);
					//将上层父亲节点 经过排序后的代表点集合的第一个代表点    加入到该节点中
					node_ij_2->add_RP(*it_rp);
					//父亲结点tempFather添加 node_ij_2为孩子结点
					tempFather->add_SonNode(node_ij_2);
				}
			}
		}
		//=========================================================================
		// 创建一个新层
		//=========================================================================
		set<Node*> Nodes;
		LayerNodes->push_back(Nodes);
		
		layer++;
		// 计算循环条件
		lastLayer = getINodesI( &(*LayerNodes)[layer-1] );
		curLayer = getINodesI( &(*LayerNodes)[layer] );
	}

}


/*
查找r_wait的邻居结点
input:
	Root: 查找树的根结点
	r_wait: 新加入的代表点
	threshold: 阈值，用于判断距离
Output:
	R_neighbor: r_wait代表点的邻居集合
	vector<Node*> *Path: 保存路径
*/
void FindingNeighbors(Node* Root, RP* r_wait, double threshold, int attrs, vector<RP*>* ret, vector<Node*> *Path)
{
	//=========================================================================
	// layer 表示当前操作 查询树的第layer层    根结点在第0层
	// similarNode 表示相似结点 similarNode[0] 表示第0层的相似结点
	// P 指向操作的结点
	//=========================================================================
	int layer = 1;
	vector<vector<Node*>*> *similarNode = new vector<vector<Node*>*>();
	Node* P = Root;//指向当前操作的结点
	Root->add_RP(r_wait);

	//=========================================================================
	// 根结点放入到 SimilarNode[0]
	//=========================================================================
	vector<Node*>* rNode = new vector<Node*>();
	rNode->push_back(Root);
	similarNode->push_back(rNode);

	//=========================================================================
	// 根结点的所有孩子放入到 SimilarNode[1]
	//=========================================================================
	vector<Node*>* sNode = new vector<Node*>();
	vector<Node*>* son = Root->sons_Node;
	vector<Node*>::iterator it;
	for (it = son->begin(); it != son->end(); it++)
	{
		sNode->push_back(*it);
	}
	similarNode->push_back(sNode);
	//=========================================================================
	// 判断树的第layer层的相似结点集合的size是否为0
	// layer 一层一层的处理
	//=========================================================================
	while ((similarNode[layer]).size() != 0)
	{
		//取出当前层的相似结点集合中需要操作的结点（在similarNode中）
		vector<Node*>* curLayerS = (*similarNode)[layer];
		// 产生一个新结点，并将r_wait存入该节点
		Node* newNode = new Node(layer, NULL);
		newNode->add_RP(r_wait);
		// 一个标志判断是否存在相似（用newNode 和 相似集合中结点 比较）
		bool notExistSimilarity = true;
		//=========================================================================
		// ( 1 )
		//=========================================================================
		//存储与newNode相似结点的孩子结点
		vector<Node*>* ChildNode = new vector<Node*>();
		for (it = curLayerS->begin(); it != curLayerS->end(); it++)
		{
			//=========================================================================
			// 计算相似性   使用Definition 5
			//=========================================================================
			Node *curNode = *it;
			//计算两个结点的相似性
			bool similarity = isSimilarityNN(curNode, newNode, layer);
			//判断相似性
			if (similarity)
				//=========================================================================
				// 处理第一、二种情况
				// 代表点和结点相似 则将代表点加入到结点包含的集合之中（newNode --融合到--> curNode）
				// 这时候是不是应该 将curNode设置为newNode？？？？？？？？？？？
				//=========================================================================
			{
				merge(curNode, newNode);
				newNode = curNode;//融合后的curNode变为新节点
				vector<Node*>* sons = curNode->sons_Node;
				vector<Node*>::iterator it_s;
				for (it_s = sons->begin(); it_s != sons->end(); it_s++)
				{
					ChildNode->push_back(*it_s);
				}
				notExistSimilarity = false;
			}
		}
		//=========================================================================
		// 第三种情况的时候 
		// 从当前结点开始构建一条独立的子树
		//=========================================================================
		if (notExistSimilarity == true)
		{//构建一条全新树的支路（从根节点开始）
			P->add_SonNode(newNode);
			while (layer < attrs)
			{
				P = newNode;
				layer++;
				Node* newNode_s = new Node(layer, NULL);
				newNode_s->add_RP(r_wait);
				//
				P->add_SonNode(newNode);
			}
			return;
			//break;
		}
		//将当前结点P指向 newNode ，并且放入路径Path中 （Path在算法四的时候会使用）
		P = newNode;
		Path->push_back(P);

		//----------???????????????????????排序结点
		sort(ChildNode->begin(), ChildNode->end(), cmpNN);

		//新的 similarNode 中的一个对象
		vector<Node*>* vNode = new vector<Node*>();
		//=========================================================================
		// ( 2 )
		//=========================================================================
		//取出存放ChildNode中第一个
		Node* pNode = ChildNode->at(0);
		Node *nextNode;
		vector<Node*>::iterator it_s;

		for (it_s = ChildNode->begin() + 1; it_s != ChildNode->end(); it_s++)
		{
			nextNode = *it_s;
			bool similarity = isSimilarityNN(pNode, nextNode, layer);
			//判断相似性
			if (similarity)
			{	//代表点和结点相似 则将代表点加入到结点包含的集合之中
				merge(pNode, nextNode);
			}
			else
			{//保证pNode 和nextNode结点的相邻
				vNode->push_back(pNode);
				pNode = nextNode;
			}
		}
		//similarNode添加一层相似结点层
		similarNode->push_back(vNode);

		//到了查询树的最后一层的时候
		if (layer == attrs)
		{
			vector<RP*> *rps = newNode->R;
			vector<RP*>::iterator it_r;
			for (it_r = rps->begin(); it_r != rps->end(); it_r++)
			{
				RP* rp = *it_r;
				if (rp != r_wait && getRPDistance(rp, r_wait, attrs) <= 2 * threshold)
				{
					ret->push_back(rp);
				}
			}
		}
		layer++;
	}
}

/*
更新聚类
	算法3在算法4中被循环调用
input:
	R: 代表点集合（来自算法1）
	G: 代表点的连接图（来自算法1）
	U2: 新加入记录矩阵
	clusters: 聚类集合
	attrs:属性数目
	alpha：
	beta:
	threshold: 阈值，用于判断距离
Output:
	clusters: 聚类集合
*/
void UpdatingClustering(vector<RP*> *R, Graph *graph, double alpha, double beta, double threshold,
	int attrs, Node* root, double **U2, Graph *G, vector<Cluster*> *clusters)
{
	vector<RP*>::iterator R_itor;
	vector<Node*>::iterator it_node;
	for (R_itor = R->begin(); R_itor != R->end(); R_itor++)
	{
		RP* r_wait = *R_itor;
		vector<RP*>* R_neighbor = new vector<RP*>();
		vector<Node*> *Path = new vector<Node*>();

		//（1）=========================================================================
		//	获取r_wait的邻居结点集合（调用算法三）
		//=========================================================================
		FindingNeighbors(root, r_wait, threshold, attrs, R_neighbor, Path);

		vector<int>::iterator it_x;
		vector<int>* cover = &(r_wait->Cover);
		//存储没有映射到R_neighbor的记录|对象
		vector<double> *noMapping = new vector<double>();
		//（2）=========================================================================
		// 用于判断 在r_wait中的记录Or对象 是否映射到 R_neighbor中的邻居代表点
		// 如果存在映射		则将r_wait中包含的对象存放到对应映射代表点 
		// 不存在映射		
		//=========================================================================
		bool flag;

		//（3）=========================================================================
		// 判断r_wait代表点中记录x 和 r_wait的邻居代表点的距离dist
		// dist  <  阈值： 则将记录x 放入到 对应的r_wait的邻居代表点 的覆盖域Cover中
		//			否则： 将记录x放入到存放无映射对象的集合noMapping中
		//=========================================================================
		vector<RP*>::iterator it_rp;
		for (it_x = cover->begin(); it_x != cover->end(); it_x++)
		{
			double x = *it_x;
			flag = true;
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				double dist = getDistanceRX(x, rp, U2, attrs);
				if (dist <= threshold)//判断条件
				{
					rp->Cover.push_back(x);
					flag = false;
				}
			}
			if (flag)
			{
				noMapping->push_back(x);
			}
		}
		//（4）=========================================================================
		// 三种情况的最后一种情况（r_wait代表点 和其他代表点的没有重合时候）
		// 判断 r_wait 和其邻居代表点中对象的距离 是否小于阈值
		// 
		//=========================================================================
		if (noMapping->size() == 0)
		{
			//遍历 r_wait 的邻居代表点
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				vector<int>* cover2 = &(rp->Cover);
				// 遍历邻居代表点中的记录(对象)
				for (it_x = cover2->begin(); it_x != cover2->end(); it_x++)
				{
					double x = *it_x;
					//计算 r_wait  与 邻居点对象之间的距离
					double dist = getDistanceRX(x, r_wait, U2, attrs);
					if (dist <= threshold)
					{
						//Cover(r_wait) = Cover(r_wait) U x
						//邻居点对象放入到 r_wait 代表点的覆盖域
						r_wait->Cover.push_back(x);
					}
				}
			}
			//将r_wait代表点 放入到 R_neighbor
			R_neighbor->push_back(r_wait);
		}
		//（5）=========================================================================
		//三种情况的前两种情况（r_wait代表点 和附近其他代表点的部分Or完全重合时候）
		//从（算法3产生）Path路径中删除 结点中的 r_wait 代表点
		//=========================================================================
		else
		{
			vector<RP*>::iterator it_rp2;
			//遍历Path路径中的查询树结点
			for (it_node = Path->begin(); it_node != Path->end(); it_node++)
			{
				Node* node = *it_node;
				//取出查询树结点的代表点集合
				vector<RP*> *R = node->R;
				for (it_rp2 = R->begin(); it_rp2 != R->end(); it_rp2++)
				{
					//删除路径中查询树结点之中的 r_wait 代表点
					if (*it_rp2 == r_wait)
					{
						R->erase(it_rp2);
						break;
					}
				}
			}
		}

		//（6）=========================================================================
		// 更新子图 ？？？？？？？？？？？？？？？？
		// 更新r_wait的邻居代表点 与其周围代表点的连接
		// ---------------------------------------------
		// 疑问   ？？？  
		// r_wait代表点 放入到 R_neighbor后 ===》代表着有新的代表点生成 （4）
		// 那么要在 G 中增加r_wait 代表点的行和列
		//=========================================================================
		double similarityRiRj;
		//遍历 r_wait 的邻居代表点  更新  
		for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
		{
			//获取 r_wait的邻居代表点 的邻居集合neighbor
			//vector<RP*> *neighbor = &((*it_rp)->rpNeighbor->Nei);
			vector<RP*> *neighbor = ((*it_rp)->rpNeighbor);
			vector<RP*>::iterator it_rp_n;
			for (it_rp_n = neighbor->begin(); it_rp_n != neighbor->end(); it_rp_n++)
			{
				RP* ri = *it_rp;
				RP* rj = *it_rp_n;
				//获取两个代表点的相似度
				similarityRiRj = computeSimilarity(ri, rj);
				if (similarityRiRj >= alpha)
				{
					//strong link
					G->addStrongLink(ri, rj);
				}
				if (similarityRiRj < alpha && similarityRiRj >= beta)
				{
					//weak link
					G->addWeakLink(ri, rj);
				}
			}
		}
	}
	//（7）=========================================================================
	// 获取最终的聚类结果
	//=========================================================================
	int newN = 1;
	list<vector<int>*>::iterator it; //声明一个迭代器 用于迭代Strong link的
	list<set<int>*>::iterator itW; //声明一个迭代器 用于迭代weak link的
	for (it = G->subGraphs.begin(), itW = G->subGraphsWeak.begin();
		it != G->subGraphs.end();
		it++, itW++) {
		Cluster *Cx = new Cluster(newN);

		for (int i = 0; i < (*it)->size(); i++)
		{
			//根据 存储的矩阵行或列中的位置      映射到    代表点的指针
			RP* rp = G->getMapping((*(*it))[i]);//(*(*it))[i]    (*it)->at(i)
			Cx->addCoverToPOS(rp);
		}
		set<int>::iterator itW2;
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//根据 存储的矩阵行或列中的位置      映射到    代表点的指针
			RP* rp = G->getMapping(*itW2);
			Cx->addCoverToBND(rp);
		}
		newN++;
		clusters->push_back(Cx);
	}
}


