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
-- �洢��¼ ��������Ϣ
*/
class X {
	double no;
	//double *val;
	vector<double> val;
public:
	//���캯��
	X(double *v, int attrs)
	{
		//val = new double[attrs];
		val.resize(attrs);
		no = v[0];
		for (int i = 0; i < attrs; i++)
			//v[i+1] ӦΪU[i][0]  ��ŵ��Ǽ�¼�ı�� U[i][ 1 �� attrs-1]��Ĳ�������ֵ 
			val[i] = v[i + 1];
	}
	X(vector<double> &v, int attrs)
	{
		//val = new double[attrs];
		val.resize(attrs);
		no = v[0];
		for (int i = 0; i < attrs; i++)
			//v[i+1] ӦΪU[i][0]  ��ŵ��Ǽ�¼�ı�� U[i][ 1 �� attrs-1]��Ĳ�������ֵ 
			val[i] = v[i + 1];
	}
	~X()
	{
		/*
		cout << no << " : ";
		for (int i = 0; i < 5; i++)
			//v[i+1] ӦΪU[i][0]  ��ŵ��Ǽ�¼�ı�� U[i][ 1 �� attrs-1]��Ĳ�������ֵ 
			cout<< "   "<< val[i];
		cout << endl;
		//delete[] val;*/
	}
	//��ȡ����ֵ
	double getVal(int i)
	{
		return val[i];
	}

};

/*
	center��¼�ھӼ���   ��    ��
	center �� ��¼�ţ����ģ�
	count ���ھ���Ŀ
	Nei �� �ھӼ���
*/
class Neighbor {
public:
	// ���� ������¼��¼���룩
	int center;
	// �ھ���Ŀ
	int count;
	// �ھӵļ���
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
-- �����Representation point
-- representationPoint : �����¼
-- Cover : ����� ������� ���󼯺�
-- left : �������Ե�����ֵ�½�
-- right : �������Ե�����ֵ�Ͻ�
*/
class RP {
public:
	//�����  
	int representationPoint;
	// ���ǵĶ��󼯺�
	vector<int> *Cover;

	//RNeighbor* rpNeighbor;
	vector<RP*> *rpNeighbor;
	//��СΪ���Ե���Ŀ  left���������½�   right ���������Ͻ�
	//double *left;
	vector<double> left;
	vector<double> right;
	//double *right;
	//Neis��Xi�ĸ�������
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
		//���ô����
		setRepresentationPoint(rp);
		//���ø��ǵ�
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

	//Cover �������
	void CoverPush(int x)
	{
		Cover->push_back(x);
	}
	//��ȡ���ǵĴ�С
	int getCoverSize()
	{
		return Cover->size();
	}

	//��ȡ���ǵ��±��Ӧ��ֵ
	int getCoverVal(int i)
	{
		return Cover->at(i);
	}

	//���������±꣨�ڼ����±꣩��ȡ����ֵ
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
	//���������±꣨�ڼ����±꣩��ȡ����ֵ
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
	//�������
	//double **distance;
	vector<vector<double>> distance;
	/*
	-- ״̬����  ���ڴ�ž����е�״̬
	-- ��Ϊ������ʹ�ù��ľ������row ����ɾ��������ʹ��state�洢ɾ��״̬
	-- ��ʼ��Ϊ 0   ɾ����Ϊ 1
	*/
	//int *state;
	vector<double> state;
	// ��Ч��¼��Ŀ  ��δ����ɾ���ļ�¼��Ŀ��
	int rows;
	// ALL��¼��Ŀ  
	int NUM;

	/*
	-- ��ʼ��������� ʹ���±� 1 - N ��λ��
	-- N����¼��Ŀ
	*/
	Distance(int N)
	{
		NUM = rows = N;
		//Ϊ ״̬���� ���ٿռ�
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

	//����ŷ�Ｘ�¾���
	double getDistance(X *x, X *y, int attrs)
	{
		// ret = sqrt( �� pow(x(i)-y(i)) )
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
	-- ���þ���������еĶ���֮����� N*N
	-- distance[][] ֻ��ʹ�� 1 �� N����ʹ�� 0 ��λ�ã�
	-- U ���������
	-- N �����������У�������Ŀ
	-- attrs :������Ŀ
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
				{	//������룬ʹ��ŷ����¾���
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
				{	//������룬ʹ��ŷ����¾���
					X *x = new X((*U)[i], attrs);
					X *y = new X((*U)[j], attrs);
					setDisVal(i, j, getDistance(x, y, attrs));
				}
			}
		}
	}
	//���� i�����м�¼��   j�����м�¼��  val��Ҫ���õ�ֵ
	void setDisVal(int i, int j, double val)
	{
		//*(*(distance + i) + j) = val;
		distance[i][j] = val;
	}
	//��ȡ i�����м�¼����   j�����м�¼��
	double getDisVal(int i, int j)
	{
		//return *(*(distance + i) + j);
		return distance[i][j];
	}

	//����״̬ i�����м�¼��  
	void setState(int i)
	{
		//*(state + i) = 1;
		state[i] = 1;
	}
	//��ȡ״̬ i�����м�¼��  
	double getState(int i)
	{
		//return *(state + i);
		return state[i];
	}
	//��ȡ��Ч�ļ�¼��Ŀ
	int getRows()
	{
		return rows;
	}
	/*ɾ����������е���   ����Ǽ�ɾ��ֻ�Ǹı�״̬����

	row �Ǿ�����к�  ���Ǽ�¼��

	*/
	void deleteRows(int row)
	{
		setState(row);
		rows--;
	}
	/*
	set the centroid of Rnew be the object x1 corresponding to the first row of Distance(x,y)

	��Rnew����������Ϊ��Distance(x,y)�ĵ�һ�ж�Ӧ�Ķ���x1��

	�����м�¼���� ��ɾ�� ��״̬   ��return -1
	*/

	int getFirstRecord(vector<Neighbor*> *neighbors)
	{
		//���������ĵ�һ��״̬Ϊ δɾ�� ��
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

//center��¼������ھӼ���   ��    ��
class RNeighbor {
public:
	// ���� ������¼��¼���룩
	RP *center;
	// �ھ���Ŀ
	int count;
	// �ھӵļ���
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
	�������������
*/
class Node {
public:
	//����п��ܴ��ڶ�������
	vector<RP*> *R;

	//���ĵڼ��� ith layer ����Ϊ��0��  ʹ������0 ���ǵ�һ��
	int ith;
	//���ith�����Ե� ���½��ֵ
	double low;
	double high;


	//�洢���ӽ��
	vector<Node*> *sons_Node;
	//�洢���׽��
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

	//��Ӻ��ӽ��
	void add_SonNode(Node* newNode)
	{
		newNode->father_Node = this;
		sons_Node->push_back(newNode);
	}
	//��ӽ������ĵĴ����
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
	//�������
	double **similarity;
	//ALL�������Ŀ  
	int NUM;

	//N����¼��Ŀ  ��ʼ��
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
	//N����¼��Ŀ  ��ʼ��
	~Similarity()
	{
		for (int j = 0; j < NUM + 1; j++)
		{
			delete[] * (similarity + j);
		}
		delete[] similarity;
	}


	//���� i�����м�¼��   j�����м�¼��  val��Ҫ���õ�ֵ
	void setSimilarVal(int i, int j, double val)
	{
		*(*(similarity + i) + j) = val;
	}
	//��ȡ i�����м�¼����   j�����м�¼��
	double getSimilarVal(int i, int j)
	{
		return *(*(similarity + i) + j);
	}

};

class Graph {
public:
	//���ͼ�ľ���
	int **G;
	//���ǿ��ͨ��ͼ
	// vector<int> �д�����ھ����л����е�λ��   ����   ӳ�䵽������ָ��
	list<vector<int>*> subGraphs;
	//���ǿ��ͨ��ͼ 
	list<set<int>*> subGraphsWeak;
	//������ֵ �� ����λ��ӳ���    ����λ��ΪKey    �����Ϊ value
	map<RP*, int> mapping_RP_Loc;
	//������ֵ �� ����λ��ӳ���     ����λ��Ϊvalue    �����Ϊ Key
	map< int, RP*> mapping_Loc_RP;
	//�������Ŀ
	int RN;
	//ʹ�ô���㼯�� ��ʼ�� ͼ���ڽӾ���
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

	//ri��rj֮������ǿ����
	void addStrongLink(RP* ri, RP* rj)
	{
		int i, j;
		i = getMapping(ri);
		j = getMapping(rj);
		*(*(G + i) + j) = STRONG_LINK;
	}

	//ri��rj֮������������
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
	//-----------------------ͼ����--------------------------------------
	/*��һ�����*/
	int nextNeighbor(int row, int cur)
	{
		//����ͼ��һ����|�����Ǿ���
		for (int i = cur + 1; i < RN; i++)
		{
			if (*(*(G + row) + i) == 2)
			{
				return i;
			}
		}
		return -1;

	}
	//�ж��Ƿ����δ���ʵĽ�㣨RP��
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
	//�洢�����ӵĴ����
	void visitOpWeak(int row, set<int> *subGW)
	{
		//����ͼ��һ����|�����Ǿ���
		for (int i = 0; i < RN; i++)
		{
			if (*(*(G + row) + i) == 1)
			{
				subGW->insert(i);
			}
		}
	}
	/*��ȡ����ǿ������ͼ*/
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
			//��ͼ�Ĵ洢
			if (subG->size() > 1)
			{
				subGraphs.push_back(subG);
				subGraphsWeak.push_back(subGW);
			}
		}
	}
	/*��ȡһ��ǿ������ͼ*/
	void subGraph_BFS(int *visited, int first, vector<int>* subGStrongLink, set<int> *subGWeakLink)
	{
		queue<int> queue;
		queue.push(first);
		//���ʽ��  visited�����Ӧλ������Ϊ �ѷ��ʵ�״̬
		visitOp(visited, first);
		//�������������   ���ҽ������ӵ������Ӵ���� ���� subGWeakLink����֮��
		visitOpWeak(first, subGWeakLink);
		subGStrongLink->push_back(first);
		while (!queue.empty())
		{
			int current = queue.front();
			queue.pop();
			for (int justVisit = nextNeighbor(current, -1);//��ȡ��һ���ھӣ��� 
				justVisit < RN && justVisit != -1; //justVisit�ھӽڵ�ķ�Χ���ɳ��� RN  ������õ�
				justVisit = nextNeighbor(current, justVisit))//��ȡ��һ���ھ�
			{
				//��û�з��ʹ��ý�㣨����㣩
				if (*(visited + justVisit) == 0)
				{
					//visited
					subGStrongLink->push_back(justVisit);
					//����ǿ�������Ĳ���
					visitOp(visited, justVisit);
					//�������������Ĳ��� 
					visitOpWeak(justVisit, subGWeakLink);
					//�������
					queue.push(justVisit);
				}
			}

		}
	}

};

class Cluster {

public:
	//�������
	int Cx;
	set<int> *POS;//��ŵ��Ǽ�¼
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

	//������㸲�����еļ�¼�������������
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

//�ھӼ����ھ���Ŀ��������
bool cmpNeighbor(const Neighbor *nFirst, const Neighbor *nSecond);
//��ȡ���ݼ�¼�Ż�ȡ ��¼���ھӼ���
vector<int>* getNeighborByRecord(vector<Neighbor*> *neighbors, int record);
/*===========================================================
-- ��������֮���ŷ�Ｘ�¾���
===========================================================*/
double getRPDistance(RP *x, RP *y, int attrs);
/*===========================================================
�����������������ƶ�
Similarity = ��  Cover(i) �� Cover(j) �� / min{Cover(i) ��Cover(j)}
===========================================================*/
double computeSimilarity(RP* ri, RP* rj);

bool isSimilarity(Node * node, RP* rp, int layer);

bool isSimilarityNN(Node * node, Node * nnode, int layer);

//��ȡ�����Ŀ
int getINodesI(set<Node*> *Nodes);

//RP��㼯�ϵ������� ʹ�ÿ�������
void QSort(vector<RP*>* ivec, vector<RP*>::iterator low, vector<RP*>::iterator high, int layer);
vector<RP*>::iterator Partition(vector<RP*>*, vector<RP*>::iterator low, 
	vector<RP*>::iterator high, int layer);
bool cmpRP_Big(RP* r1, RP* r2, int layer);
bool cmpRP_Small(RP* r1, RP* r2, int layer);

//####################################################################################
//	�ĸ��㷨������
//####################################################################################

/*
U ��¼����
N ��¼��Ŀ
attrs ������Ŀ
alpha
beta
threshold ��ֵ
*/
void SOC_TWD(vector<vector<double>> *U, int N, int attrs, double alpha, double beta, double threshold,
	vector<RP*> *R, Graph **Graph2, vector<Cluster*> **clusters2);

/*
Ĭ�ϲ���ֻ����������ʱ��д����
*/
void CST(vector<RP*> *R, int attrs, Node *Root, double threshold = 0.9);

void FindingNeighbors(Node* Root, RP* r_wait, double threshold, 
	int attrs, vector<RP*>* ret, vector<Node*> *Path);

void UpdatingClustering(vector<RP*> *R, double alpha, double beta, double threshold,
	int attrs, Node* root, vector<vector<double>> *U2, Graph *G, vector<Cluster*> *clusters);


//####################################################################################
//  ��ӡ����������������ʾ��
//####################################################################################

void printU(vector<vector<double>> *U);
//��ӡ�ھӼ���
void printNei(const vector<Neighbor*> *neis);

//��ӡ����㼯��
void printR(const vector<RP*> *R, int attrs);

//��ӡ��¼״̬����
void printState(Distance *distance);
//��ӡ�����de ͼ
void printG(Graph *graph);

//��ӡǿ������ͼ �� ��������ͼ
void printGR(Graph *graph);

void printCluster(vector<Cluster*> *clusters);

void printNodeR(vector<RP*> *node_R, int layer);

void printTree(Node* root);