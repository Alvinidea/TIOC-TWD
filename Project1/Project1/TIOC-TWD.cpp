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
vector<int>* getNeighborByRecord(vector<Neighbor*> *neighbors, int record)
{
	int flag = 0, i = 0;
	for (; i < neighbors->size(); i++)
	{
		if ((*neighbors)[i]->center == record)
		{
			Neighbor* temp = (*neighbors)[i];
			flag = 1;
			return temp->getNei();
		}
	}
	if (flag == 0)
		return NULL;
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
	set<int> ri_cover ;
	set<int> rj_cover;
	vector<int>::iterator insert_iterator;
	set<int> res;//存储交集结果

	double min, inter, isize, jsize;
	double ret;
	int temp;
	for (int i = 0; i < ri->getCoverSize(); i++)
	{
		temp = ri->getCoverVal(i);
		ri_cover.insert(temp);
		res.insert(temp);
	}
	for (int j = 0; j < rj->getCoverSize(); j++)
	{
		temp = rj->getCoverVal(j);
		rj_cover.insert(temp);
		res.insert(temp);
	}
	jsize = rj_cover.size();
	isize = ri_cover.size();
	min = isize < jsize ? isize : jsize;
	//
	//std::set_intersection(ri_cover->begin(), ri_cover->end(), ri_cover->begin(), ri_cover->end(), res->begin());
	inter = jsize+ isize - res.size();
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

//-----------------------------------------------------------------

bool cmpRP_Big(RP* r1, RP* r2, int layer)
{
	double l1 = r1->getLeft(layer);
	double l2 = r2->getLeft(layer);
	return (l1 >= l2) ? true : false;
}
bool cmpRP_Small(RP* r1, RP* r2, int layer)
{
	double l1 = r1->getLeft(layer);
	double l2 = r2->getLeft(layer);
	return (l1 <= l2) ? true : false;
}
vector<RP*>::iterator Partition(vector<RP*>*, vector<RP*>::iterator low,
	vector<RP*>::iterator high, int layer)
{
	vector<RP*>::value_type pivokey = *low;
	while (low < high)
	{
		while (low < high && cmpRP_Small(pivokey, *high, layer)) high--; // 若此处为 *high >= pivokey; 则对于5 8 5，进行快速排序仍然为 5 8 5， 不能将
		// 序列变为 5 5 8. 且最后high和low均为第一个元素，故QSort中iter-1会出现越界错误;
		*low = *high;
		while (low < high && cmpRP_Big(pivokey, *low, layer)) low++;
		*high = *low;
		/*
		while (low < high && *high > pivokey) high--; // 若此处为 *high >= pivokey; 则对于5 8 5，进行快速排序仍然为 5 8 5， 不能将
		// 序列变为 5 5 8. 且最后high和low均为第一个元素，故QSort中iter-1会出现越界错误;
		*low = *high;
		while (low < high && *low <= pivokey) low++;
		*high = *low;
		*/
	}
	*low = pivokey;
	return low;
}
void QSort(vector<RP*>* ivec, vector<RP*>::iterator low, vector<RP*>::iterator high, int layer)
{
	if (low < high)
	{
		vector<RP*>::iterator iter = Partition(ivec, low, high, layer);
		if(low != iter)
			QSort(ivec, low, iter - 1, layer);
		if(high != iter)
		QSort(ivec, iter + 1, high, layer);
	}
}
//-----------------------------------------------------------------


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
bool isSimilarityNR(Node * node, RP* rp, int layer)
{
	double rp_left = rp->getLeft(layer);
	double rp_right = rp->getRight(layer);
	double node_left = node->low;
	double node_right = node->high;
	if (rp_left > node_right || node_left > rp_right)
		return false;
	return true;
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
	vector<RP*>::iterator it_r = R->begin();
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
			f_s->erase(it_son);//从父结点中删除 newNode 结点
			flag = true;
			break;
		}
	}
	//孩子结点的融合后需要重新 结点计算结点的上下界
	treeNode->computeLowAndHigh(treeNode->ith - 1);
}

//联合
bool isSimilarityNN(Node * node, Node * nnode, int layer)
{
	RP *rp = nnode->R->at(0);
	return isSimilarity(node, rp, layer);
}
bool isSimilarityNN(Node * curNode, Node * nNode)
{
	if (curNode->low > nNode->high || curNode->high < nNode->low)
		return false;
	else
		return true;
}

//结点的比较函数
bool cmpNN(Node *r1, Node *r2)
{//STL 中 >大于 时候 返回true时候  为降序
	//STL 中 <小于 时候 返回true时候  为升序
	if (r1->low < r2->low)
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
double getDistanceRX(int rd, RP* rp, vector<vector<double>> *U, int attrs)
{
	double ret, sum, temp;
	int j = rp->representationPoint;//取出代表点的中心
	X x = X((*U)[rd - 1], attrs);
	X y = X((*U)[j - 1], attrs);
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

void printU(vector<vector<double>> *U)
{
	for (int i = 0; i < U->size(); i++)
	{
		for (int j = 0; j < 6; j++)
			cout << (*U)[i][j] << '\t';
		cout << endl;
	}
}
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
		cout << '\t' << "邻居数目：" << nei->count << "\t::  ";
		vector<int> *Nei = nei->getNei();
		for (int j = 0; j < Nei->size(); j++)
		{
			cout << (*Nei)[j] << " , ";
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
		cout <<"### "<< i+1 <<":  代表点：" << rp->representationPoint << endl;
		cout << '\t' << "覆盖域：";
		for (int j = 0; j < rp->getCoverSize(); j++)
		{
			cout << (*(rp->Cover))[j] << " , ";
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
	for (int i = 0; i < distance->NUM ; i++)
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
			int sign = *(*((graph->G) + i) + j);
			switch (sign)
			{
			case 0:
				cout << GREEN <<sign << "  ";
				break;
			case 1:
				cout << WHITE << sign << "  ";
				break;
			case 2:
				cout << RED << sign << "  ";
				break;
			}
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

		cout << "\n\t" << "第" << itr << "个弱关联子图：" << '\t';
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

void printNodeR(vector<RP*> *node_R, int layer)
{
	cout << "==========================================" << endl;
	cout << "第 "<<layer <<" 层结点上下界" << endl;
	cout << "==========================================" << endl;
	vector<RP*>::iterator it;
	for (it = node_R->begin(); it != node_R->end(); it++)
	{
		RP* rp = *it;
		cout << "<" << rp->getLeft(layer) << "，" << rp->getRight(layer) << ">" << endl;
	}
}

void printTree(Node *root)
{
	bool flag = true;
	cout << "==========================================" << endl;
	cout << "打印查找树" << endl;
	cout << "==========================================" << endl;
	cout << "\t 结点顺序<孩子数目>{代表点数目}" << endl 
		<< "\t=========================================="<<"\n\t";
	int pnode = 0;
	int level, front, rear,last;
	level = 0;
	front = rear =-1;
	last = 0;
	vector<Node*>::iterator it;
	queue<Node*> que;
	que.push(root);
	rear++;
	while (que.size() != 0)
	{
		Node* node = que.front();

		que.pop();
		front++;
		if(flag)
			cout << RED << pnode <<"<"<< node->sons_Node->size()<<">"
			<< "{" << node->R->size() << "}"
			<< ",";
		else
			cout << GREEN << pnode <<"<" << node->sons_Node->size() << ">" 
			<< "{" << node->R->size() << "}" << ",";
		flag = true;//用于打印
		pnode++;
		if (node->sons_Node != NULL && node->sons_Node->size() != 0)
		{

			for (it = node->sons_Node->begin(); it != node->sons_Node->end(); it++)
			{
				que.push(*it); 
				rear++;
			}
			flag = false;
		}
		if (front == last)
		{
			level++;
			last = rear;
			cout << endl << endl<<'\t';
		}
	}
	cout << "==========================================" << endl;
	cout << GREEN;
}

void printNode(vector<Node*> *node)
{
	if (node->size() == 0)
		return;
	cout << "==========================================" << endl;
	cout << "第 " << node->at(0)->ith - 1 << " 层结点上下界" << endl;
	cout << "==========================================" << endl;
	vector<Node*>::iterator it;
	for (it = node->begin(); it != node->end(); it++)
	{
		Node* node = *it;
		cout << "<" << node->low << "，" << node->high << ">" << endl;
	}
}

//=================================================================================================
//####################################################################################
//  四大算法
//####################################################################################
//=================================================================================================

/*
算法1：
	U 记录矩阵
	N 记录数目
	attrs 属性数目
	alpha
	beta
	threshold 阈值
*/
void SOC_TWD(vector<vector<double>> *U, int N, int attrs, double alpha, double beta, double threshold,
	vector<RP*> *R, Graph **Graph2, vector<Cluster*> **clusters2)
{
	//----------------------------------------------------------------------------
	//--初始化
	//----------------------------------------------------------------------------
	// R 存放代表点的容器
	//vector<RP*> *R = new vector<RP*>();
	// neighbors 存放各个记录的所有邻居的容器
	vector<Neighbor*> *neighbors = new vector<Neighbor*>();
	// distance 存放两个记录之间距离的矩阵
	Distance *distance = new Distance(N);

	//vector<RNeighbor*> rneighbors;
	vector<vector<RP*>*> rneighbors;

	/*----------------------------------------------------------------------------
	--计算距离矩阵 计算每个对象（记录）之间的距离
	----------------------------------------------------------------------------*/
	distance->setAllDistance(U, N, attrs);
	/*----------------------------------------------------------------------------
	-- 计算Xi的邻居
	-- 所有的记录的邻居都要计算
	----------------------------------------------------------------------------*/
	for (int i = 0; i < N; i++)
	{
		Neighbor *nei = new Neighbor(i);
		for (int j = 0; j < N; j++)
		{
			if (i == j)			//邻居不需要计算自身
				;
			else
			{//判断两个对象的距离是否 <= threshold
				if (distance->getDisVal(i, j) <= threshold)
				{//小于等于则加入到该对象的邻居集中
					nei->addNei(j);
				}
			}
		}
		//存储 Xi 的邻居数目
		nei->count = nei->getNeiSize();
		// 存入所有点的邻居集
		neighbors->push_back(nei);
	}
	/*----------------------------------------------------------------------------
	-- 对neighbors排序
	-- 按每个Xi（对象|记录）的邻居数目降序排序
	----------------------------------------------------------------------------*/
	printNei(neighbors);
	std::sort(neighbors->begin(), neighbors->end(), cmpNeighbor);
	printNei(neighbors);
	/*----------------------------------------------------------------------------
	-- 创建 R 代表点集合,并将所有代表点加入进去
	-- RP 包括 代表点   覆盖区域   属性上 和 下界
	----------------------------------------------------------------------------*/
	//判断 距离矩阵 中是否还存在Row（以下while循环中会有假删除操作）
	while (distance->getRows() != 0)
	{
		//选择第一行（距离矩阵 未删除数据的第一行）
		int record = distance->getFirstRecord(neighbors);
		if (record == -1)//说明已经删完了
			break;
		// 存放邻居集合
		vector<int> *nei = getNeighborByRecord(neighbors, record);
		// 获取nei  根据记录号 在 存放所有记录邻居集合 中获取对应邻居集合

		//生成代表点 设置代表记录  和  覆盖区域   初始化属性值上下界
		RP *rp = new RP(record, &nei, attrs);
		rp->setLeftAndRight(U, attrs);
		//将中心点也加入到覆盖域中
		rp->CoverPush(record);
		for (int i = 0; i < rp->getCoverSize(); i++)
		{
			int reco = (*(rp->Cover))[i];
			//设置代表点的 属性上|下界
			for (int attr = 0; attr < attrs; attr++)
			{
				//Cover中保存的是对象号  +1第一行存的是记录号
				// U 是从 0 开始计算   而对象号从 1 开始算 ，所以*(U + reco -1)
				//double attrVal = *(*(U + reco) + 1 + attr);
				double attrVal = (*U)[reco][1 + attr];
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
		vector<RP*> *rnei = (ri->rpNeighbor);
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


	*Graph2 = G;
	*clusters2 = clusters;
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
void CST(vector<RP*> *R,int attrs,Node *Root, double threshold)
{
	/*
		vector<set<Node*>>					存放所有层的 结点集合
		set<Node*>			Node(i)			某一层的结点集合
		Node				node			一个结点
	*/
	vector<set<Node*>> *LayerNodes = new vector<set<Node*>>();
	int layer = 0;
	// 根结点的父结点为NULL
	//Node *Root = new Node(layer,NULL);
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
		// 创建一个新层
		//=========================================================================
		set<Node*> Nodes;
		LayerNodes->push_back(Nodes);
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
			vector<RP*> *node_R = (*it_node_ij)->R;
			// 对上界还是下界排序 ？ 第几个属性 ？ 
			// 下界left 第layer个属性
			vector<RP*>::iterator low = node_R->begin();
			vector<RP*>::iterator high = node_R-> end();
			high = high - 1;
			QSort(node_R, low, high, layer);
			printNodeR(node_R,layer);
			//std::sort(R->begin(),R->end(), cmpRP);
			
			//=========================================================================
			//	创建第 i+1 层新结点（设置它的层数 和 父亲结点）
			//	将上层父亲节点 经过排序后的代表点集合的 第一个 代表点    加入到该节点中
			//=========================================================================
			Node *node_ij_1 = new Node(layer + 1, tempFather);
			tempFather->add_SonNode( node_ij_1 );//父亲结点添加孩子结点
			node_ij_1->add_RP( *(node_R->begin()) );
			//=========================================================================
			//	通过代表点的相似度对第 i+1 层结点进行划分
			//=========================================================================
			vector<RP*>::iterator it_rp;
			bool similarity = true;
			Node *now_Node = node_ij_1;//当前操作的结点
			for (it_rp = node_R->begin() + 1; it_rp != node_R->end(); it_rp++)
			{
				//=========================================================================
				// 计算相似性   使用Definition 5
				//=========================================================================
				//similarity = isSimilarity(now_Node, *it_rp,layer);
				now_Node->computeLowAndHigh(layer);
				similarity = isSimilarityNR(now_Node, *it_rp, layer);
				//判断相似性
				if (similarity)
				{	//代表点和结点相似 则将代表点加入到结点包含的集合之中
					now_Node->add_RP(*it_rp);
				}
				else//代表点和结点不相似 则创建新的结点
				{
					//=========================================================================
					// 计算该结点（now_Node）的 的第layer属性的low 和 high值
					// 并插入到树的第 layer 层
					//=========================================================================
					now_Node->computeLowAndHigh( layer );
					// 插入到查找树的第 layer+1 层
					((*LayerNodes)[layer + 1]).insert(now_Node);
					//=========================================================================
					// 创建一个新的结点 设置它的层数 和 父亲结点
					// 并设置其包含的初始代表点
					//=========================================================================
					Node *node_ij_2 = new Node(layer + 1, tempFather);
					//将上层父亲节点 经过排序后的代表点集合的第一个代表点    加入到该节点中
					node_ij_2->add_RP(*it_rp);
					tempFather->add_SonNode(node_ij_2);//父亲结点tempFather添加 node_ij_2为孩子结点
					now_Node = node_ij_2;//将当前操作结点修改为 node_ij_2
				}
			}
			if (similarity)
			{
				//=========================================================================
					// 计算该结点（now_Node）的 的第layer属性的low 和 high值
					// 并插入到树的第 layer 层
					//=========================================================================
				now_Node->computeLowAndHigh(layer);
				// 插入到查找树的第layer层
				((*LayerNodes)[layer + 1]).insert(now_Node);
			}
			else 
			{
				vector<Node*>::iterator finalson = tempFather->sons_Node->end() - 1;
				(*finalson)->computeLowAndHigh(layer);
				// 插入到查找树的第layer层
				((*LayerNodes)[ layer + 1 ]).insert(*finalson);
				//node_ij_2->computeLowAndHigh(layer);
			}
		}
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
void FindingNeighbors(Node* Root, RP* r_wait, double threshold, int attrs, 
	vector<RP*>* retR_nei, vector<Node*> *Path)
{
	printTree(Root);
	//=========================================================================
	// layer 表示当前操作 查询树的第layer层    根结点在第0层
	// similarNode 表示相似结点 similarNode[0] 表示第0层的相似结点
	// P 指向操作的结点
	//=========================================================================
	//保存 相似结点构成的层（相似层） 的容器
	vector<vector<Node*>*> *similarNode = new vector<vector<Node*>*>();
	Node* P = Root;//指向当前操作的结点
	Root->add_RP(r_wait);
	Path->push_back(P);

	//=========================================================================
	// 根结点放入到 SimilarNode[0]
	//=========================================================================
	vector<Node*>* rNode = new vector<Node*>();//生成根结点层
	rNode->push_back(Root);	//将根结点压入该层
	similarNode->push_back(rNode);//将 相似层0 放入 容器

	//=========================================================================
	// 根结点的所有孩子放入到 SimilarNode[1]
	//=========================================================================
	vector<Node*>* sNode = new vector<Node*>();//存放根结点的孩子的 相似层1
	vector<Node*>* son = Root->sons_Node;
	vector<Node*>::iterator it;
	for (it = son->begin(); it != son->end(); it++)
	{
		sNode->push_back(*it);
	}
	similarNode->push_back(sNode);//将 相似层1 放入容器
	//=========================================================================
	// 判断树的第layer层的相似结点集合的size是否为0
	// layer 一层一层的处理
	//=========================================================================
	int similarLayer = 1;	//第 0 层是根结点所在的层，从第 1 层开始(相似层)
	int attrlayer = 0;
	while ((similarNode[similarLayer]).size() != 0)
	{
		//取出当前层的相似结点集合中需要操作的结点（在similarNode中）
		vector<Node*>* curLayerS = (*similarNode)[similarLayer];
		Node* newNode = new Node(similarLayer, NULL);// 产生一个新结点
		newNode->add_RP(r_wait);//将r_wait加入到新节点中
		newNode->computeLowAndHigh(attrlayer);//计算结点的Low和High值
		bool existSimilarity = false;// 一个判断是否存在相似的标志（用newNode 和 相似集合中结点 比较）
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
			//计算两个结点的相似性？？？？？？？？？？？？？？？？？
			//bool similarity = isSimilarityNN(curNode, newNode, attrlayer);
			bool similarity = isSimilarityNN(curNode,newNode);
			//判断相似性
			if (similarity)
				//=========================================================================
				// 处理第一、二种情况
				// 代表点和结点相似 则将代表点加入到结点包含的集合之中（newNode --融合到--> curNode）
				// 这时候是不是应该 将curNode设置为newNode？   ：是的
				//=========================================================================
			{
				merge(curNode, newNode);//将newNode融合到curNode中 融合后 
										// 需要重新计算结点上下界(merge内部会操作)
				newNode = curNode;		//融合后的curNode变为新节点
				vector<Node*>* sons = curNode->sons_Node;
				vector<Node*>::iterator it_s;
				//将 curNode 的所有孩子结点放到ChildNode中
				for (it_s = sons->begin(); it_s != sons->end(); it_s++)
				{
					ChildNode->push_back(*it_s);
				}
				existSimilarity = true;//代表存在相似结点
			}
		}
		//=========================================================================
		// 第三种情况的时候 
		// 从当前结点开始构建一条独立的子树
		// 若进入到此 if 语句中 那么执行完毕后可以直接跳出本函数
		//=========================================================================
		if (existSimilarity == false)
		{//构建一条全新树的支路（从当前节点开始）
			P->add_SonNode(newNode);
			while (attrlayer < attrs)
			{
				P = newNode;
				attrlayer++;
				Node* newNode_s = new Node(attrlayer, NULL);
				newNode_s->add_RP(r_wait);
				P->add_SonNode(newNode);
			}
			return;
			//break;
		}
		//=========================================================================
		// 路径 和 排序 
		//=========================================================================
		P = newNode;		//将当前结点P指向 newNode
		Path->push_back(P);	//并且放入路径Path中 （Path在算法四的时候会使用）
		//printNode(ChildNode);
		std::sort(ChildNode->begin(), ChildNode->end(), cmpNN);// 排序结点 使用升序
		//printNode(ChildNode);
		vector<Node*>* simiNodes = new vector<Node*>();// 新的 similarNode 中的一个对象
		//=========================================================================
		// ( 2 ) 产生下一个循环的 相似结点层
		//=========================================================================
		//取出存放ChildNode中第一个结点

		if (ChildNode->size() != 0)
		{
			Node* new_Node = ChildNode->at(0);
			Node *pNode;
			vector<Node*>::iterator it_s;//迭代器
			// (1) (2) (3) (4) (5) (6) (7) ; 
			// pNode=(1)  nextNode = (2)
			// pNode , nextNode 若融合(相似)，则pNode = (1,2)，nextNode=(3)
			// pNode , nextNode 若不融合(不相似)，则pNode = (2)，nextNode=(3)
			for (it_s = ChildNode->begin() + 1; it_s != ChildNode->end(); it_s++)
			{
				pNode = *it_s;
				//bool similarity = isSimilarityNN(pNode, nextNode, attrlayer);
				bool similarity = isSimilarityNN(new_Node, pNode);
				//判断相似性
				if (similarity)
				{	//将pNode融合到newNode中
					// merge操作 融合的是树中的结点   不会影响到ChildNode
					merge(new_Node,pNode);//。。。。。。。。。。。。。。。。。。。。。。。。。。。。
				}
				else
				{//保证 pNode 和 nextNode 结点的相邻
					simiNodes->push_back(new_Node);//将 ChildNode中结点 加入到 新的相似层
					new_Node = pNode;//
				}
			}
			//将 ChildNode中结点 加入到 新的相似层
			simiNodes->push_back(new_Node);
			//similarNode添加一层相似结点层
			similarNode->push_back(simiNodes);
		}
		//=========================================================================
		// ( 4 ) 到了查询树的最后一层的时候
		// 最后计算 newNode结点中的代表点中心和r_wait的距离 
		// 将其保存在r_wait的邻居代表点集合中
		//=========================================================================
		if (similarLayer == attrs)
		{
			vector<RP*> *rps = newNode->R;	//获取 newNode 包含的代表点集合
			vector<RP*>::iterator it_r;		//迭代器
			for (it_r = rps->begin(); it_r != rps->end(); it_r++)
			{
				RP* rp = *it_r;
				if (rp != r_wait && getRPDistance(rp, r_wait, attrs) <= 2 * threshold)
				{
					retR_nei->push_back(rp);//
				}
			}
			break;
		}
		else {
			similarLayer++;
			attrlayer++;
		}
	}

	printTree(Root);
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
void UpdatingClustering(vector<RP*> *R, double alpha, double beta, double threshold,
	int attrs, Node* root, vector<vector<double>> *U, Graph *G, vector<Cluster*> *clusters)
{
	vector<RP*>::iterator R_itor;
	vector<Node*>::iterator it_node;
	for (R_itor = R->begin(); 
		//R_itor != R->end();
		R_itor != R->begin() + 3;
		R_itor++)
	{
		//-------------------------------------------------------------------????????????
		RP* r_wait = *R_itor;							//取出代表点指针
		vector<RP*>* R_neighbor = new vector<RP*>();	//r_wait 代表点的邻居代表点集合
		vector<Node*> *Path = new vector<Node*>();
		//-------------------------------------------------------------------????????????
		//（1）=========================================================================
		//	获取r_wait的邻居结点集合（调用算法三）
		//=========================================================================
		
		FindingNeighbors(root, r_wait, threshold, attrs, R_neighbor, Path);
		cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
		vector<int>::iterator it_x;		//迭代器 用于迭代r_wait的覆盖域的对象
		vector<int>* cover = (r_wait->Cover);
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
		for (it_x = cover->begin(); it_x != cover->end(); 
			it_x++)
		{//r_wait的覆盖域的对象: it_x
			double x = *it_x;
			flag = true;//
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{//r_wait 的邻居代表点: rp
				RP* rp = *it_rp;
				double dist = getDistanceRX(x, rp, U, attrs);//获取距离
				if (dist <= threshold)//判断条件
				{
					rp->CoverPush(x);
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
		// 判断 r_wait 和其邻居代表点中对象x 的距离 是否小于阈值
		// 小于 阈值 时候，将x放到r_wait覆盖域中去
		//=========================================================================
		if (noMapping->size() == 0)
		{
			//遍历 r_wait 的邻居代表点
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				vector<int>* cover2 = (rp->Cover);
				// 遍历邻居代表点中的记录(对象)
				for (it_x = cover2->begin(); it_x != cover2->end(); it_x++)
				{
					double x = *it_x;
					//计算 r_wait  与 邻居点对象之间的距离
					double dist = getDistanceRX(x, r_wait, U, attrs);
					if (dist <= threshold)
					{
						//Cover(r_wait) = Cover(r_wait) U x
						//邻居点对象放入到 r_wait 代表点的覆盖域
						r_wait->CoverPush(x);
					}
				}
			}
			//将r_wait代表点 放入到 R_neighbor
			R_neighbor->push_back(r_wait);
		}
		//（5）=========================================================================
		//	三种情况的前两种情况（r_wait代表点 和附近其他代表点的部分Or完全重合时候）
		//	从（算法3产生）Path路径中删除 结点中的 r_wait 代表点
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
		vector<RP*> *neighbor;
		vector<RP*>::iterator it_rp_n;
		for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
		{
			//获取 r_wait的邻居代表点 的邻居集合neighbor
			neighbor = ((*it_rp)->rpNeighbor);
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
	printG(G);
	printGR(G);
	printCluster(clusters);
}

//=================================================================================================