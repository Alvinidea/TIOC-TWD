#pragma once
#include<set>
#include<map>
#include<list>
#include<queue>
#include<vector>
#include<math.h>
#include <iterator>
#include<algorithm>
#include<iostream>
using namespace std;

#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */


#define MAX 99999
#define MIN -99999

#define STRONG_LINK 2
#define WEAK_LINK 1
/*
-- 存储记录 的属性信息
*/
class X {
	double no;
	//double *val;
	vector<double> val;
public:
	//构造函数
	X(double *v, int attrs)
	{
		//val = new double[attrs];
		val.resize(attrs);
		no = v[0];
		for (int i = 0; i < attrs; i++)
			//v[i+1] 应为U[i][0]  存放的是记录的编号 U[i][ 1 到 attrs-1]存的才是属性值 
			val[i] = v[i + 1];
	}
	X(vector<double> &v, int attrs)
	{
		//val = new double[attrs];
		val.resize(attrs);
		no = v[0];
		for (int i = 0; i < attrs; i++)
			//v[i+1] 应为U[i][0]  存放的是记录的编号 U[i][ 1 到 attrs-1]存的才是属性值 
			val[i] = v[i + 1];
	}
	~X()
	{
		/*
		cout << no << " : ";
		for (int i = 0; i < 5; i++)
			//v[i+1] 应为U[i][0]  存放的是记录的编号 U[i][ 1 到 attrs-1]存的才是属性值 
			cout<< "   "<< val[i];
		cout << endl;
		//delete[] val;*/
	}
	//获取属性值
	double getVal(int i)
	{
		return val[i];
	}

};

/*
	center记录邻居集合   的    类
	center ： 记录号（中心）
	count ：邻居数目
	Nei ： 邻居集合
*/
class Neighbor {
public:
	// 中心 （本记录记录号码）
	int center;
	// 邻居数目
	int count;
	// 邻居的集合
	//vector<int> Nei;
	vector<int> *Nei;

	Neighbor(int cente)
	{
		count = 0;
		center = cente;
		Nei = new vector<int>();
	}
	~Neighbor()
	{
		delete Nei;
	}
	void addNei(int newNeighbor)
	{
		Nei->push_back(newNeighbor);
		//count++;
	}
	int getNeiSize()
	{
		return Nei->size();
	}

	vector<int>* getNei()
	{
		return Nei;
	}
};

/*
-- 代表点Representation point
-- representationPoint : 代表记录
-- Cover : 代表点 所代表的 对象集合
-- left : 各个属性的属性值下界
-- right : 各个属性的属性值上界
*/
class RP {
public:
	//代表点  
	int representationPoint;
	// 覆盖的对象集合
	vector<int> *Cover;

	//RNeighbor* rpNeighbor;
	vector<RP*> *rpNeighbor;
	//大小为属性的数目  left代表属性下界   right 代表属性上界
	//double *left;
	vector<double> left;
	vector<double> right;
	//double *right;
	//Neis是Xi的覆盖区域
	RP(int rp, vector<int> **Neis, int attrs)
	{
		//left = new double[attrs];
		//right = new double[attrs];
		left.resize(attrs);
		right.resize(attrs);
		for (int i = 0; i < attrs; i++)
		{
			//*(left + i) = MAX;
			//*(right + i) = MIN;
			left[i] = MAX;
			right[i] = MIN;
		}
		//设置代表点
		setRepresentationPoint(rp);
		//设置覆盖点
		setCover(Neis);

		//rpNeighbor = new RNeighbor(this);
		rpNeighbor = new vector<RP*>();
	}

	~RP()
	{
		delete Cover;
		delete rpNeighbor;
	}
	void setLeftAndRight(double *U[],int attrs)
	{
		double temp = 0;
		for (int i = 0; i < attrs; i++)
		{
			temp = *(*(U + representationPoint) + i+1);
			right[i] = temp;
			left[i] = temp;
		}
	}
	void setLeftAndRight(vector<vector<double>> *U, int attrs)
	{
		double temp = 0;
		for (int i = 0; i < attrs; i++)
		{
			temp = (*U)[representationPoint][i+1];
			//temp = *(*(U + representationPoint) + i + 1);
			right[i] = temp;
			left[i] = temp;
		}
	}
	void setRepresentationPoint(int x)
	{
		representationPoint = x;
	}

	void setCover(vector<int> **x)
	{
		//Cover->assign((*x)->begin(), (*x)->end());
		Cover = *x;
	}

	//Cover 添加数据
	void CoverPush(int x)
	{
		Cover->push_back(x);
	}
	//获取覆盖的大小
	int getCoverSize()
	{
		return Cover->size();
	}

	//获取覆盖的下标对应的值
	int getCoverVal(int i)
	{
		return Cover->at(i);
	}

	//根据属性下标（第几个下标）获取属性值
	double getLeft(int i)
	{
		//return *(left + i);
		return left[i];
	}

	double getRight(int i)
	{
		//return *(right + i);
		return right[i];
	}
	//根据属性下标（第几个下标）获取属性值
	void setLeft(int i, double val)
	{
		//*(left + i) = val;
		left[i] = val;
	}
	void setRight(int i, double val)
	{
		//*(right + i) = val;
		right[i] = val;
	}
};

class Distance {
public:
	//距离矩阵
	//double **distance;
	vector<vector<double>> distance;
	/*
	-- 状态数组  用于存放矩阵行的状态
	-- 因为后面会对使用过的距离矩阵row 进行删除操作，使用state存储删除状态
	-- 初始化为 0   删除后为 1
	*/
	//int *state;
	vector<double> state;
	// 有效记录数目  （未被假删除的记录数目）
	int rows;
	// ALL记录数目  
	int NUM;

	/*
	-- 初始化距离矩阵 使用下标 1 - N 的位置
	-- N：记录数目
	*/
	Distance(int N)
	{
		NUM = rows = N;
		//为 状态矩阵 开辟空间
		//state = new int(N + 1);
		//distance = new double*[N + 1];
		distance.resize(NUM);
		state.resize(NUM);
		for (int i = 0; i < N ; i++)
		{
			//*(state + i) = 0;
			//*(distance + i) = new double[N + 1];
			distance[i].resize(NUM);
			for (int j = 0; j < N ; j++)
			{
				distance[i][j] = 0;
				//*(*(distance + i) + j) = 0;
			}
		}
	}

	~Distance()
	{	}

	//计算欧里几德距离
	double getDistance(X *x, X *y, int attrs)
	{
		// ret = sqrt( ∑ pow(x(i)-y(i)) )
		double ret, sum, temp;
		ret = sum = temp = 0;
		for (int i = 0; i < attrs; i++)
		{
			temp = x->getVal(i) - y->getVal(i);
			temp = pow(temp, 2);
			sum += temp;
		}
		ret = sqrt(sum);
		return ret;
	}

	/*
	-- 设置距离矩阵所有的对象之间距离 N*N
	-- distance[][] 只是使用 1 到 N（不使用 0 号位置）
	-- U ：对象矩阵
	-- N ：对象矩阵的行（对象）数目
	-- attrs :属性数目
	*/
	void setAllDistance(double *U[], int N, int attrs)
	{
		for (int i = 0; i < N ; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (i == j)
					;
				else
				{	//计算距离，使用欧几里德距离
					X *x = new X(*(U + i ), attrs);
					X *y = new X(*(U + j ), attrs);
					setDisVal(i, j, getDistance(x, y, attrs));
				}
			}
		}
	}
	void setAllDistance(vector<vector<double>> *U, int N, int attrs)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				if (i == j)
					;
				else
				{	//计算距离，使用欧几里德距离
					X *x = new X((*U)[i], attrs);
					X *y = new X((*U)[j], attrs);
					setDisVal(i, j, getDistance(x, y, attrs));
				}
			}
		}
	}
	//设置 i代表行记录号   j代表列记录号  val是要设置的值
	void setDisVal(int i, int j, double val)
	{
		//*(*(distance + i) + j) = val;
		distance[i][j] = val;
	}
	//获取 i代表行记录号码   j代表列记录号
	double getDisVal(int i, int j)
	{
		//return *(*(distance + i) + j);
		return distance[i][j];
	}

	//设置状态 i代表行记录号  
	void setState(int i)
	{
		//*(state + i) = 1;
		state[i] = 1;
	}
	//获取状态 i代表行记录号  
	double getState(int i)
	{
		//return *(state + i);
		return state[i];
	}
	//获取有效的记录数目
	int getRows()
	{
		return rows;
	}
	/*删除距离矩阵中的行   这儿是假删除只是改变状态而已

	row 是矩阵的行号  就是记录号

	*/
	void deleteRows(int row)
	{
		setState(row);
		rows--;
	}
	/*
	set the centroid of Rnew be the object x1 corresponding to the first row of Distance(x,y)

	将Rnew的质心设置为与Distance(x,y)的第一行对应的对象x1。

	若所有记录都是 已删除 的状态   则return -1
	*/

	int getFirstRecord(vector<Neighbor*> *neighbors)
	{
		//返回遇到的第一个状态为 未删除 的
		vector<Neighbor*>::iterator it;
		for (it = neighbors->begin(); it != neighbors->end(); it++)
		{
			//if (*(state + i) == 0)
			int row = (*it)->center;
			if (state[row] == 0)
			{	
				deleteRows(row);
				return row;
			}
		}
		return -1;
	}
};

//center记录代表点邻居集合   的    类
class RNeighbor {
public:
	// 中心 （本记录记录号码）
	RP *center;
	// 邻居数目
	int count;
	// 邻居的集合
	vector<RP*> Nei;


	RNeighbor(RP * cente)
	{
		count = 0;
		center = cente;
	}
	~RNeighbor()
	{
		delete center;
	}

	void addNeighbor(RP* newNeighbor)
	{
		Nei.push_back(newNeighbor);
		count++;
	}
};

/*
	搜索树的树结点
*/
class Node {
public:
	//结点中可能存在多个代表点
	vector<RP*> *R;

	//树的第几层 ith layer ，根为第0层  使用属性0 的是第一层
	int ith;
	//存放ith个属性的 上下界的值
	double low;
	double high;


	//存储孩子结点
	vector<Node*> *sons_Node;
	//存储父亲结点
	Node* father_Node;


	Node(int layer, Node* father)
	{
		ith = layer;
		sons_Node = new vector<Node*>();
		father_Node = father;
		R = new vector<RP*>();
	}

	~Node()
	{
		delete R;
		delete sons_Node;
		delete father_Node;
	}

	//添加孩子结点
	void add_SonNode(Node* newNode)
	{
		newNode->father_Node = this;
		sons_Node->push_back(newNode);
	}
	//添加结点包含的的代表点
	void add_RP(RP* rp)
	{
		R->push_back(rp);
	}

	void computeLowAndHigh(int layer)
	{
		vector<RP*>::iterator it_rp;
		RP * first = *(R->begin());
		double temp_low, temp_high;
		low = first->getLeft(layer);
		high = first->getRight(layer);
		for (it_rp = R->begin() + 1; it_rp != R->end(); it_rp++)
		{
			temp_low = (*it_rp)->getLeft(layer);
			temp_high = (*it_rp)->getRight(layer);
			if (low > temp_low)
				low = temp_low;
			if (high < temp_high)
				high = temp_high;
		}
	}
};

class Similarity
{
	//距离矩阵
	double **similarity;
	//ALL代表点数目  
	int NUM;

	//N：记录数目  初始化
	Similarity(int N)
	{
		NUM = N;
		similarity = new double*[N + 1];
		for (int i = 0; i < N + 1; i++)
		{
			*(similarity + i) = new double[N + 1];
			for (int j = 0; j < N + 1; j++)
			{
				*(*(similarity + i) + j) = 0;
			}
		}
	}
	//N：记录数目  初始化
	~Similarity()
	{
		for (int j = 0; j < NUM + 1; j++)
		{
			delete[] * (similarity + j);
		}
		delete[] similarity;
	}


	//设置 i代表行记录号   j代表列记录号  val是要设置的值
	void setSimilarVal(int i, int j, double val)
	{
		*(*(similarity + i) + j) = val;
	}
	//获取 i代表行记录号码   j代表列记录号
	double getSimilarVal(int i, int j)
	{
		return *(*(similarity + i) + j);
	}

};

class Graph {
public:
	//存放图的矩阵
	int **G;
	//存放强连通子图
	// vector<int> 中存的是在矩阵行或列中的位置   可以   映射到代表点的指针
	list<vector<int>*> subGraphs;
	//存放强连通子图 
	list<set<int>*> subGraphsWeak;
	//代表点的值 到 矩阵位置映射表    矩阵位置为Key    代表点为 value
	map<RP*, int> mapping_RP_Loc;
	//代表点的值 到 矩阵位置映射表     矩阵位置为value    代表点为 Key
	map< int, RP*> mapping_Loc_RP;
	//代表点数目
	int RN;
	//使用代表点集合 初始化 图（邻接矩阵）
	Graph(vector<RP*> *R)
	{
		RN = R->size();
		for (int i = 0; i < RN; i++)
		{
			addMapping(R->at(i), i);
			addMapping(i, R->at(i));
		}
		G = new int*[RN];
		for (int i = 0; i < RN; i++)
		{
			*(G + i) = new int[RN];
			for (int j = 0; j < RN; j++)
			{
				*(*(G + i) + j) = 0;
			}
		}
	}

	~Graph()
	{
		for (int j = 0; j < RN; j++)
		{
			delete[] * (G + j);
		}
		delete[] G;
	}

	//ri和rj之间设置强连接
	void addStrongLink(RP* ri, RP* rj)
	{
		int i, j;
		i = getMapping(ri);
		j = getMapping(rj);
		*(*(G + i) + j) = STRONG_LINK;
	}

	//ri和rj之间设置弱连接
	void addWeakLink(RP* ri, RP* rj)
	{
		int i, j;
		i = getMapping(ri);
		j = getMapping(rj);
		*(*(G + i) + j) = WEAK_LINK;
	}

	int getMapping(RP* key)
	{
		return mapping_RP_Loc.at(key);
	}
	void addMapping(RP* key, int val)
	{
		auto pair = std::make_pair(key, val);
		mapping_RP_Loc.insert(pair);
	}

	RP* getMapping(int key)
	{
		return mapping_Loc_RP.at(key);
	}
	void addMapping(int key, RP* val)
	{
		auto pair = std::make_pair(key, val);
		mapping_Loc_RP.insert(pair);
	}
	//-----------------------图操作--------------------------------------
	/*下一个结点*/
	int nextNeighbor(int row, int cur)
	{
		//无向图是一个上|下三角矩阵
		for (int i = cur + 1; i < RN; i++)
		{
			if (*(*(G + row) + i) == 2)
			{
				return i;
			}
		}
		return -1;

	}
	//判断是否存在未访问的结点（RP）
	int isExistVisitedRP(int *visited)
	{
		for (int i = 0; i < RN; i++)
		{
			if (*(visited + i) == 0)
			{
				return i;
			}
		}
		return -1;

	}

	void visitOp(int *visited, int i)
	{
		*(visited + i) = 1;
	}
	//存储弱连接的代表点
	void visitOpWeak(int row, set<int> *subGW)
	{
		//无向图是一个上|下三角矩阵
		for (int i = 0; i < RN; i++)
		{
			if (*(*(G + row) + i) == 1)
			{
				subGW->insert(i);
			}
		}
	}
	/*获取所有强连接子图*/
	void BFS()
	{
		int *visited = new int[RN];
		for (int i = 0; i < RN; i++)
			*(visited + i) = 0;
		int notVisited;
		while ((notVisited = isExistVisitedRP(visited)) != -1)
		{
			vector<int> *subG = new vector<int>();
			set<int> *subGW = new set<int>();
			subGraph_BFS(visited, notVisited, subG, subGW);
			//子图的存储
			if (subG->size() > 1)
			{
				subGraphs.push_back(subG);
				subGraphsWeak.push_back(subGW);
			}
		}
	}
	/*获取一个强连接子图*/
	void subGraph_BFS(int *visited, int first, vector<int>* subGStrongLink, set<int> *subGWeakLink)
	{
		queue<int> queue;
		queue.push(first);
		//访问结点  visited数组对应位置设置为 已访问的状态
		visitOp(visited, first);
		//访问弱关联结点   并且将相连接的弱连接代表点 放入 subGWeakLink容器之中
		visitOpWeak(first, subGWeakLink);
		subGStrongLink->push_back(first);
		while (!queue.empty())
		{
			int current = queue.front();
			queue.pop();
			for (int justVisit = nextNeighbor(current, -1);//获取第一个邻居（） 
				justVisit < RN && justVisit != -1; //justVisit邻居节点的范围不可超过 RN  访问完该点
				justVisit = nextNeighbor(current, justVisit))//获取下一个邻居
			{
				//若没有访问过得结点（代表点）
				if (*(visited + justVisit) == 0)
				{
					//visited
					subGStrongLink->push_back(justVisit);
					//访问强关联结点的操作
					visitOp(visited, justVisit);
					//访问弱关联结点的操作 
					visitOpWeak(justVisit, subGWeakLink);
					//进入队列
					queue.push(justVisit);
				}
			}

		}
	}

};

class Cluster {

public:
	//聚类号码
	int Cx;
	set<int> *POS;//存放的是记录
	set<int> *BND;

	Cluster(int i)
	{
		Cx = i;
		POS = new set<int>();
		BND = new set<int>();
	}

	~Cluster()
	{
		delete POS;
		delete BND;
	}

	RP* getRPbyNum(int key, Graph *G)
	{
		return G->getMapping(key);
	}

	//将代表点覆盖域中的记录放入聚类正域中
	void addCoverToPOS(RP* rp)
	{
		for (int i = 0; i < rp->getCoverSize(); i++)
		{
			POS->insert(rp->getCoverVal(i));
		}
	}

	void addCoverToBND(RP* rp)
	{
		for (int i = 0; i < rp->getCoverSize(); i++)
		{
			BND->insert(rp->getCoverVal(i));
		}
	}
};

//####################################################################################
//####################################################################################
//####################################################################################
//####################################################################################

//邻居集的邻居数目降序排序
bool cmpNeighbor(const Neighbor *nFirst, const Neighbor *nSecond);
//获取根据记录号获取 记录的邻居集合
vector<int>* getNeighborByRecord(vector<Neighbor*> *neighbors, int record);
/*===========================================================
-- 计算代表点之间的欧里几德距离
===========================================================*/
double getRPDistance(RP *x, RP *y, int attrs);
/*===========================================================
计算两个代表点的相似度
Similarity = 【  Cover(i) 交 Cover(j) 】 / min{Cover(i) ，Cover(j)}
===========================================================*/
double computeSimilarity(RP* ri, RP* rj);

bool isSimilarity(Node * node, RP* rp, int layer);

bool isSimilarityNN(Node * node, Node * nnode, int layer);

//获取结点数目
int getINodesI(set<Node*> *Nodes);

//RP结点集合的排序函数 使用快速排序
void QSort(vector<RP*>* ivec, vector<RP*>::iterator low, vector<RP*>::iterator high, int layer);
vector<RP*>::iterator Partition(vector<RP*>*, vector<RP*>::iterator low, 
	vector<RP*>::iterator high, int layer);
bool cmpRP_Big(RP* r1, RP* r2, int layer);
bool cmpRP_Small(RP* r1, RP* r2, int layer);

//####################################################################################
//	四个算法的声明
//####################################################################################

/*
U 记录矩阵
N 记录数目
attrs 属性数目
alpha
beta
threshold 阈值
*/
void SOC_TWD(vector<vector<double>> *U, int N, int attrs, double alpha, double beta, double threshold,
	vector<RP*> *R, Graph **Graph2, vector<Cluster*> **clusters2);

/*
默认参数只需在声明的时候写出来
*/
void CST(vector<RP*> *R, int attrs, Node *Root, double threshold = 0.9);

void FindingNeighbors(Node* Root, RP* r_wait, double threshold, 
	int attrs, vector<RP*>* ret, vector<Node*> *Path);

void UpdatingClustering(vector<RP*> *R, double alpha, double beta, double threshold,
	int attrs, Node* root, vector<vector<double>> *U2, Graph *G, vector<Cluster*> *clusters);


//####################################################################################
//  打印方法（用于数据显示）
//####################################################################################

void printU(vector<vector<double>> *U);
//打印邻居集合
void printNei(const vector<Neighbor*> *neis);

//打印代表点集合
void printR(const vector<RP*> *R, int attrs);

//打印记录状态矩阵
void printState(Distance *distance);
//打印代表点de 图
void printG(Graph *graph);

//打印强关联子图 和 弱关联子图
void printGR(Graph *graph);

void printCluster(vector<Cluster*> *clusters);

void printNodeR(vector<RP*> *node_R, int layer);

void printTree(Node* root);