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

#define MAX 99999
#define MIN -99999

#define STRONG_LINK 2
#define WEAK_LINK 1
/*
-- �洢��¼ ��������Ϣ
*/
class X {
	double no;
	double *val;
public:
	//���캯��
	X(double *v, int attrs)
	{
		val = (double*)malloc(sizeof(double)*attrs);
		no = v[0];
		for (int i = 0; i < attrs; i++)
			//v[i+1] ӦΪU[i][0]  ��ŵ��Ǽ�¼�ı�� U[i][1 �� attrs-1]��Ĳ�������ֵ 
			val[i] = v[i + 1];
	}
	//��ȡ����ֵ
	double getVal(int i)
	{
		return val[i];
	}

};

class Distance {
public:
	//�������
	double **distance;
	/*
	-- ״̬����  ���ڴ�ž����е�״̬
	-- ��Ϊ������ʹ�ù��ľ������row ����ɾ��������ʹ��state�洢ɾ��״̬
	-- ��ʼ��Ϊ 0   ɾ����Ϊ 1
	*/
	int *state;
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
		state = new int(N + 1);
		distance = new double*[N + 1];
		for (int i = 0; i < N + 1; i++)
		{
			*(state + i) = 0;
			*(distance + i) = new double[N + 1];
			for (int j = 0; j < N + 1; j++)
			{
				*(*(distance + i) + j) = 0;
			}
		}
	}

	//����ŷ�Ｘ�¾���
	double getDistance(X x, X y, int attrs)
	{
		// ret = sqrt( �� pow(x(i)-y(i)) )
		double ret, sum, temp;
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

	/*
	-- ���þ���������еĶ���֮����� N*N
	-- distance[][] ֻ��ʹ�� 1 �� N����ʹ�� 0 ��λ�ã�
	-- U ���������
	-- N �����������У�������Ŀ
	-- attrs :������Ŀ
	*/
	void setAllDistance(double *U[], int N, int attrs)
	{
		for (int i = 1; i < N + 1; i++)
		{
			for (int j = 1; j < N + 1; j++)
			{
				if (i == j)
					;
				else
				{	//������룬ʹ��ŷ����¾���
					X x = X(*(U + i - 1), attrs);
					X y = X(*(U + j - 1), attrs);
					setDisVal(i, j, getDistance(x, y, attrs));
				}
			}
		}
	}

	//���� i�����м�¼��   j�����м�¼��  val��Ҫ���õ�ֵ
	void setDisVal(int i, int j, double val)
	{
		*(*(distance + i) + j) = val;
	}
	//��ȡ i�����м�¼����   j�����м�¼��
	double getDisVal(int i, int j)
	{
		return *(*(distance + i) + j);
	}

	//����״̬ i�����м�¼��  
	void setState(int i)
	{
		*(state + i) = 1;
	}
	//��ȡ״̬ i�����м�¼��  
	double getState(int i)
	{
		return *(state + i);
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
	int getFirstRecord()
	{
		//���������ĵ�һ��״̬Ϊ δɾ�� ��
		for (int i = 1; i < NUM + 1; i++)
		{
			if (*(state + i) == 0)
			{
				deleteRows(i);
				return i;
			}
		}
		return -1;
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
	vector<int> Nei;


	Neighbor(int cente)
	{
		count = 0;
		center = cente;
	}
	void addNeighbor(int newNeighbor)
	{
		Nei.push_back(newNeighbor);
		//count++;
	}
};

//center��¼�ھӼ���   ��    ��
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
	void addNeighbor(RP* newNeighbor)
	{
		Nei.push_back(newNeighbor);
		count++;
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
	vector<int> Cover;

	//��СΪ���Ե���Ŀ  left���������½�   right ���������Ͻ�
	double *left;

	double *right;

	RNeighbor *rpNeighbor;
	//Neis��Xi�ĸ�������
	RP(int rp, vector<int> Neis, int attrs)
	{
		left = (double *)malloc(sizeof(double)*attrs);
		right = (double *)malloc(sizeof(double)*attrs);


		for (int i = 0; i < attrs; i++)
		{
			*(left + i) = MAX;
			*(right + i) = MIN;
		}
		//���ô����
		setRepresentationPoint(rp);
		//���ø��ǵ�
		setCover(Neis);

		rpNeighbor = new RNeighbor(this);
	}

	void setRepresentationPoint(int x)
	{
		representationPoint = x;
	}

	void setCover(vector<int> &x)
	{
		Cover = x;
	}

	//��ȡ���ǵĴ�С
	int getCoverSize()
	{
		return Cover.size();
	}

	//��ȡ���ǵĴ�С
	int getCoverVal(int i)
	{
		return Cover.at(i);
	}

	//���������±꣨�ڼ����±꣩��ȡ����ֵ
	double getLeft(int i)
	{
		return *(left + i);
	}

	double getRight(int i)
	{
		return *(right + i);
	}
	//���������±꣨�ڼ����±꣩��ȡ����ֵ
	void setLeft(int i, double val)
	{
		*(left + i) = val;
	}
	void setRight(int i, double val)
	{
		*(right + i) = val;
	}
};

/*
	�������������
*/
class Node {
public:
	//����п��ܴ��ڶ�������
	vector<RP*> *R;

	//�ڼ�������   ����˵�ڼ��� ith layer  Or  ith Attribute
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
			subGraphs.push_back(subG);
			subGraphsWeak.push_back(subGW);
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
//	�㷨1 2 �Ĺ�����������
//####################################################################################
//�ھӼ����ھ���Ŀ��������
bool cmpNeighbor(const Neighbor *nFirst, const Neighbor *nSecond);
//��ȡ���ݼ�¼�Ż�ȡ ��¼���ھӼ���
void getNeighborByRecord(vector<Neighbor*> &neighbors, int record, vector<int>& nei);
/*
-- ��������֮���ŷ�Ｘ�¾���
*/
double getRPDistance(RP *x, RP *y, int attrs);
/*
�����������������ƶ�
Similarity = ��  Cover(i) �� Cover(j) �� / min{Cover(i) ��Cover(j)}
*/
double computeSimilarity(RP* ri, RP* rj);
//####################################################################################
//�㷨3  4 �����㷨������
//####################################################################################
bool isSimilarity(Node * node, RP* rp, int layer);

bool isSimilarityNN(Node * node, Node * nnode, int layer);

//��ȡ�����Ŀ
int getINodesI(set<Node*> *Nodes);
//RP���ıȽϺ���
bool cmpRP(RP *r1, RP *r2);



//####################################################################################
//	�㷨1 2 �Ĺ�������ʵ��
//####################################################################################

//�ھӼ����ھ���Ŀ��������
bool cmpNeighbor(const Neighbor *nFirst, const Neighbor *nSecond)
{
	if (nFirst->count > nSecond->count) //�ɴ�С���� ע�� == ʱ��  ���뷵��false
	{
		return true;
	}
	return false;
}

//��ȡ���ݼ�¼�Ż�ȡ ��¼���ھӼ���
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
-- ��������֮���ŷ�Ｘ�¾���
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
�����������������ƶ�
Similarity = ��  Cover(i) �� Cover(j) �� / min{Cover(i) ��Cover(j)}
*/
double computeSimilarity(RP* ri, RP* rj)
{
	set<int> *ri_cover = new set<int>();
	set<int> *rj_cover = new set<int>();
	vector<int>::iterator insert_iterator;
	set<int> *res = new set<int>();//�洢�������

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
//�㷨3  4 �����㷨��ʵ��
//####################################################################################

//��ȡ�����Ŀ
int getINodesI(set<Node*> *Nodes)
{
	return Nodes->size();
}
//RP���ıȽϺ���
bool cmpRP(RP *r1, RP *r2)
{
	if (r1->getLeft(1) > r2->getLeft(1))
		return true;
	return false;
}

/*
function���ж�һ�������ĵ�layer������  �Ƿ���Node�еĴ��������
	rp�������
	node�����
	layer���ڼ��� �������Ϊ��0�㣩
*/
bool isSimilarity(Node * node, RP* rp, int layer)
{
	double rp_left = rp->getLeft(layer);
	double rp_right = rp->getRight(layer);
	vector<RP*> *R = node->R;
	vector<RP*>::iterator it;
	for (it = R->begin(); it != R->end(); it++)
	{
		/*����
		rp ��layer�����Ե��½�ֵ  ����  node�����д����ĵ�layer�����Ե��½�ֵ
		����Ϊ�Ź����ˣ�
		�����ж�  ��A��rp��layer�����Ե��½�ֵ  ��  ��B��node�����д����ĵ�layer�����Ե��Ͻ�ֵ �ͺ�
		�����B��>��A��  ˵�������Ʒ��� true
		*/
		if (((*it)->right)[layer] > rp_left)
			return true;
	}
	return false;
}


/*
	�� newNode��� �ںϵ� treeNode���
	��������ںϺ�
	�����ڵ㲻��,���Ǹ��׽���Ӧ���ӽ��newNode��Ҫɾ��
	������㼯���ں�
	�����ӽ���ں�
	�����½����¼���
*/
void merge(Node* treeNode, Node* newNode)
{
	//���¼������½�
	if (treeNode->low > newNode->low)
		treeNode->low = newNode->low;
	if (treeNode->high < newNode->high)
		treeNode->high = newNode->high;
	vector<RP*> *R = newNode->R;
	vector<RP*>::iterator it_r;
	//�������ں�
	for (it_r = R->begin(); it_r != R->end(); it_r++)
		treeNode->add_RP(*it_r);

	//���ӽ����ں�
	vector<Node*> *sons_Node = newNode->sons_Node;
	vector<Node*>::iterator it_s;
	for (it_s = sons_Node->begin(); it_s != sons_Node->end(); it_s++)
		treeNode->add_SonNode(*it_s);
	//ɾ��father���ĺ��ӽ��newNode
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

//����
bool isSimilarityNN(Node * node, Node * nnode, int layer)
{
	RP *rp = nnode->R->at(0);
	return isSimilarity(node, rp, layer);
}

//���ıȽϺ���
bool cmpNN(Node *r1, Node *r2)
{
	if (r1->low > r2->low)
		return true;
	return false;
}

/*
Input:
	R:	����㼯��
	G:
	alpha:
	beta:
	threshold: ��ֵ
Output:
	C:	����
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
void SOC_TWD(double *U[], int N, int attrs, double alpha, double beta, double threshold);

void CST(vector<RP*> *R, int *A, int attrs, double threshold = 0.9);

void FindingNeighbors(Node* Root, RP* r_wait, double threshold, 
	int attrs, vector<RP*>* ret, vector<Node*> *Path);

void UpdatingClustering(vector<RP*> *R, Graph *graph,
	double alpha, double beta, double threshold,
	int attrs, Node* root, double **U2, Graph *G, vector<Cluster*> *clusters);


//####################################################################################
//  ��ӡ����������������ʾ��
//####################################################################################

//��ӡ�ھӼ���
void printNei(const vector<Neighbor*> *neis)
{
	cout << "==========================================" << endl;
	cout <<"��ӡ�ھӼ��ϡ�������������" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < neis->size(); i++)
	{
		Neighbor* nei = (*neis)[i];
		cout << "���ĵ㣺" << nei->center << " ; ";
		cout << '\t' << "�ھ���Ŀ��" << nei->count << " ; ";
		for (int j = 0; j < nei->Nei.size(); j++)
		{
			cout << (nei->Nei)[j] << " , ";
		}
		cout << endl;
	}
}

//��ӡ����㼯��
void printR(const vector<RP*> *R,int attrs)
{
	cout << "==========================================" << endl;
	cout << "��ӡ����㼯�ϡ�������������" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < R->size(); i++)
	{
		RP* rp = (*R)[i];
		cout <<"����㣺"<< rp->representationPoint << endl;
		cout << '\t'<< "������";
		for (int j = 0; j < rp->Cover.size(); j++)
		{
			cout << (rp->Cover)[j] <<" , ";
		}
		cout << endl <<'\t'<<"�����½磺";
		for (int j = 0; j < attrs; j++)
		{
			cout << (rp->left)[j] << '\t';
		}
		cout << endl << '\t' << "�����Ͻ磺";
		for (int j = 0; j < attrs; j++)
		{
			cout << (rp->right)[j] << '\t';
		}
		cout << endl;
	}
}

//��ӡ��¼״̬����
void printState(Distance *distance)
{
	cout << "==========================================" << endl;
	cout << "��ӡ��¼״̬����:" << endl;
	cout << "==========================================" << endl;
	for (int i = 1; i < distance->NUM + 1; i++)
		cout << distance->state[i] << "  ";
	cout << endl;
}
//��ӡ�����de ͼ
void printG(Graph *graph)
{
	cout << "==========================================" << endl;
	cout << "��ӡ���������ɵ� Strong AND Weakͼ:" << endl;
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

//��ӡǿ������ͼ �� ��������ͼ
void printGR(Graph *graph)
{
	cout << "==========================================" << endl;
	cout << "��ӡǿ������ͼ �� ��������ͼ" << endl;
	cout << "==========================================" << endl;
	// vector<int> �д�����ھ����л����е�λ��   ����   ӳ�䵽������ָ��
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
		cout << "��"<<itr<<"��ǿ������ͼ��"<<'\t';
		for (int j = 0; j < (*it)->size(); j++)
		{
			cout << graph->getMapping((*(*it))[j])->representationPoint << "  ";
		}

		cout << '\t' << "��" << itr << "����������ͼ��" << '\t';
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//���� �洢�ľ����л����е�λ��      ӳ�䵽    ������ָ��
			cout << graph->getMapping(*itW2)->representationPoint << "  ";
		}
		cout << endl;
	}
}

void printCluster(vector<Cluster*> *clusters)
{
	cout << "==========================================" << endl;
	cout << "��ӡ����" << endl;
	cout << "==========================================" << endl;
	vector<Cluster*>::iterator it;
	int itr = 0;
	for (it = clusters->begin(); it != clusters->end(); it++)
	{
		itr++;
		cout << "��" << itr << "�����ࣺ" << '\t';
		Cluster* cl = *it;
		set<int>::iterator itPos;
		for (itPos = cl->POS->begin(); itPos != cl->POS->end(); itPos++)
		{
			cout << *itPos << '\t';
		}
		cout << endl;
	}
}