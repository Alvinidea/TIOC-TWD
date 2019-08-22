#include"initStructure.h"
using namespace std;


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
	set<int> ri_cover ;
	set<int> rj_cover;
	vector<int>::iterator insert_iterator;
	set<int> res;//�洢�������

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
//�㷨3  4 �����㷨��ʵ��
//####################################################################################
//��ȡ�����Ŀ
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
		while (low < high && cmpRP_Small(pivokey, *high, layer)) high--; // ���˴�Ϊ *high >= pivokey; �����5 8 5�����п���������ȻΪ 5 8 5�� ���ܽ�
		// ���б�Ϊ 5 5 8. �����high��low��Ϊ��һ��Ԫ�أ���QSort��iter-1�����Խ�����;
		*low = *high;
		while (low < high && cmpRP_Big(pivokey, *low, layer)) low++;
		*high = *low;
		/*
		while (low < high && *high > pivokey) high--; // ���˴�Ϊ *high >= pivokey; �����5 8 5�����п���������ȻΪ 5 8 5�� ���ܽ�
		// ���б�Ϊ 5 5 8. �����high��low��Ϊ��һ��Ԫ�أ���QSort��iter-1�����Խ�����;
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
	vector<RP*>::iterator it_r = R->begin();
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
			f_s->erase(it_son);//�Ӹ������ɾ�� newNode ���
			flag = true;
			break;
		}
	}
	//���ӽ����ںϺ���Ҫ���� ������������½�
	treeNode->computeLowAndHigh(treeNode->ith - 1);
}

//����
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

//���ıȽϺ���
bool cmpNN(Node *r1, Node *r2)
{//STL �� >���� ʱ�� ����trueʱ��  Ϊ����
	//STL �� <С�� ʱ�� ����trueʱ��  Ϊ����
	if (r1->low < r2->low)
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
double getDistanceRX(int rd, RP* rp, vector<vector<double>> *U, int attrs)
{
	double ret, sum, temp;
	int j = rp->representationPoint;//ȡ������������
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
//  ��ӡ����������������ʾ��
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
//��ӡ�ھӼ���
void printNei(const vector<Neighbor*> *neis)
{
	cout << "==========================================" << endl;
	cout << "��ӡ�ھӼ��ϡ�������������" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < neis->size(); i++)
	{
		Neighbor* nei = (*neis)[i];
		cout << "���ĵ㣺" << nei->center << " ; ";
		cout << '\t' << "�ھ���Ŀ��" << nei->count << "\t::  ";
		vector<int> *Nei = nei->getNei();
		for (int j = 0; j < Nei->size(); j++)
		{
			cout << (*Nei)[j] << " , ";
		}
		cout << endl;
	}
}

//��ӡ����㼯��
void printR(const vector<RP*> *R, int attrs)
{
	cout << "==========================================" << endl;
	cout << "��ӡ����㼯�ϡ�������������" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < R->size(); i++)
	{
		RP* rp = (*R)[i];
		cout <<"### "<< i+1 <<":  ����㣺" << rp->representationPoint << endl;
		cout << '\t' << "������";
		for (int j = 0; j < rp->getCoverSize(); j++)
		{
			cout << (*(rp->Cover))[j] << " , ";
		}
		cout << endl << '\t' << "�����½磺";
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
	for (int i = 0; i < distance->NUM ; i++)
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
		cout << "��" << itr << "��ǿ������ͼ��" << '\t';
		for (int j = 0; j < (*it)->size(); j++)
		{
			cout << graph->getMapping((*(*it))[j])->representationPoint << "  ";
		}

		cout << "\n\t" << "��" << itr << "����������ͼ��" << '\t';
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

void printNodeR(vector<RP*> *node_R, int layer)
{
	cout << "==========================================" << endl;
	cout << "�� "<<layer <<" �������½�" << endl;
	cout << "==========================================" << endl;
	vector<RP*>::iterator it;
	for (it = node_R->begin(); it != node_R->end(); it++)
	{
		RP* rp = *it;
		cout << "<" << rp->getLeft(layer) << "��" << rp->getRight(layer) << ">" << endl;
	}
}

void printTree(Node *root)
{
	bool flag = true;
	cout << "==========================================" << endl;
	cout << "��ӡ������" << endl;
	cout << "==========================================" << endl;
	cout << "\t ���˳��<������Ŀ>{�������Ŀ}" << endl 
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
		flag = true;//���ڴ�ӡ
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
	cout << "�� " << node->at(0)->ith - 1 << " �������½�" << endl;
	cout << "==========================================" << endl;
	vector<Node*>::iterator it;
	for (it = node->begin(); it != node->end(); it++)
	{
		Node* node = *it;
		cout << "<" << node->low << "��" << node->high << ">" << endl;
	}
}

//=================================================================================================
//####################################################################################
//  �Ĵ��㷨
//####################################################################################
//=================================================================================================

/*
�㷨1��
	U ��¼����
	N ��¼��Ŀ
	attrs ������Ŀ
	alpha
	beta
	threshold ��ֵ
*/
void SOC_TWD(vector<vector<double>> *U, int N, int attrs, double alpha, double beta, double threshold,
	vector<RP*> *R, Graph **Graph2, vector<Cluster*> **clusters2)
{
	//----------------------------------------------------------------------------
	//--��ʼ��
	//----------------------------------------------------------------------------
	// R ��Ŵ���������
	//vector<RP*> *R = new vector<RP*>();
	// neighbors ��Ÿ�����¼�������ھӵ�����
	vector<Neighbor*> *neighbors = new vector<Neighbor*>();
	// distance ���������¼֮�����ľ���
	Distance *distance = new Distance(N);

	//vector<RNeighbor*> rneighbors;
	vector<vector<RP*>*> rneighbors;

	/*----------------------------------------------------------------------------
	--���������� ����ÿ�����󣨼�¼��֮��ľ���
	----------------------------------------------------------------------------*/
	distance->setAllDistance(U, N, attrs);
	/*----------------------------------------------------------------------------
	-- ����Xi���ھ�
	-- ���еļ�¼���ھӶ�Ҫ����
	----------------------------------------------------------------------------*/
	for (int i = 0; i < N; i++)
	{
		Neighbor *nei = new Neighbor(i);
		for (int j = 0; j < N; j++)
		{
			if (i == j)			//�ھӲ���Ҫ��������
				;
			else
			{//�ж���������ľ����Ƿ� <= threshold
				if (distance->getDisVal(i, j) <= threshold)
				{//С�ڵ�������뵽�ö�����ھӼ���
					nei->addNei(j);
				}
			}
		}
		//�洢 Xi ���ھ���Ŀ
		nei->count = nei->getNeiSize();
		// �������е���ھӼ�
		neighbors->push_back(nei);
	}
	/*----------------------------------------------------------------------------
	-- ��neighbors����
	-- ��ÿ��Xi������|��¼�����ھ���Ŀ��������
	----------------------------------------------------------------------------*/
	printNei(neighbors);
	std::sort(neighbors->begin(), neighbors->end(), cmpNeighbor);
	printNei(neighbors);
	/*----------------------------------------------------------------------------
	-- ���� R ����㼯��,�������д��������ȥ
	-- RP ���� �����   ��������   ������ �� �½�
	----------------------------------------------------------------------------*/
	//�ж� ������� ���Ƿ񻹴���Row������whileѭ���л��м�ɾ��������
	while (distance->getRows() != 0)
	{
		//ѡ���һ�У�������� δɾ�����ݵĵ�һ�У�
		int record = distance->getFirstRecord(neighbors);
		if (record == -1)//˵���Ѿ�ɾ����
			break;
		// ����ھӼ���
		vector<int> *nei = getNeighborByRecord(neighbors, record);
		// ��ȡnei  ���ݼ�¼�� �� ������м�¼�ھӼ��� �л�ȡ��Ӧ�ھӼ���

		//���ɴ���� ���ô����¼  ��  ��������   ��ʼ������ֵ���½�
		RP *rp = new RP(record, &nei, attrs);
		rp->setLeftAndRight(U, attrs);
		//�����ĵ�Ҳ���뵽��������
		rp->CoverPush(record);
		for (int i = 0; i < rp->getCoverSize(); i++)
		{
			int reco = (*(rp->Cover))[i];
			//���ô����� ������|�½�
			for (int attr = 0; attr < attrs; attr++)
			{
				//Cover�б�����Ƕ����  +1��һ�д���Ǽ�¼��
				// U �Ǵ� 0 ��ʼ����   ������Ŵ� 1 ��ʼ�� ������*(U + reco -1)
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
	-- ����Gͼ   ����   �������ƶ�
	-- ʹ�ô���� �� ��ͼ�еĽ��
	//--------------------------------------------------------------------------*/
	Graph *G = new Graph(R);
	for (int i = 0; i < R->size(); i++)
	{
		RP* ri = (*R)[i];
		double similarityRiRj;//�洢���ƶ�
		//RNeighbor* rnei = ri->rpNeighbor;
		vector<RP*> *rnei = (ri->rpNeighbor);
		for (int j = 0; j < R->size(); j++)
		{
			RP* rj = R->at(j);
			if (i == j)
				;
			else
			{
				//��ô����֮��ľ���
				double distance = getRPDistance(ri, rj, attrs);
				if (distance <= 2 * threshold)
				{
					rnei->push_back(rj);
				}
				//����������֮������ƶ�
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
	// ��ȡǿ��ͨ��ͼ
	// ����������ͼ
	//----------------------------------------------------------------------------
	G->BFS();
	printGR(G);
	//----------------------------------------------------------------------------
	//----------------------���ɾ��༯��------------------------------------------
	//----------------------------------------------------------------------------
	vector<Cluster*> *clusters = new vector<Cluster*>();
	list<vector<int>*>::iterator it; //����һ�������� ���ڵ���Strong link��
	list<set<int>*>::iterator itW; //����һ�������� ���ڵ���weak link��
	int newN = 1;
	for (it = G->subGraphs.begin(), itW = G->subGraphsWeak.begin();
		it != G->subGraphs.end();
		it++, itW++) {
		Cluster *Cx = new Cluster(newN);

		for (int i = 0; i < (*it)->size(); i++)
		{
			//���� �洢�ľ����л����е�λ��      ӳ�䵽    ������ָ��
			RP* rp = G->getMapping((*(*it))[i]);//(*(*it))[i]    (*it)->at(i)
			Cx->addCoverToPOS(rp);
		}
		set<int>::iterator itW2;
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//���� �洢�ľ����л����е�λ��      ӳ�䵽    ������ָ��
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
�㷨2��
Creating the searching tree
	:ʹ�ò���ķ�ʽ����searching tree
input ��
	R������㼯��
	A������
	threshold����ֵ
output��
	root��root of researching tree
*/
void CST(vector<RP*> *R,int attrs,Node *Root, double threshold)
{
	/*
		vector<set<Node*>>					������в�� ��㼯��
		set<Node*>			Node(i)			ĳһ��Ľ�㼯��
		Node				node			һ�����
	*/
	vector<set<Node*>> *LayerNodes = new vector<set<Node*>>();
	int layer = 0;
	// �����ĸ����ΪNULL
	//Node *Root = new Node(layer,NULL);
	Root->R = R;
	double lastLayer, curLayer;

	// ����һ���µĲ�  ��0��
	set<Node*> Node0;
	Node0.insert(Root);
	LayerNodes->push_back(Node0);

	lastLayer = 0.1;
	curLayer = getINodesI( &Node0 );

	Node *tempFather;
	while (lastLayer/curLayer < threshold || layer < attrs)
	{
		//=========================================================================
		// ����һ���²�
		//=========================================================================
		set<Node*> Nodes;
		LayerNodes->push_back(Nodes);
		//=========================================================================
		// ������ i ���е����н��(һ����� �� ������������)
		//=========================================================================
		set<Node*>::iterator it_node_ij;
		for ( it_node_ij = LayerNodes->at(layer).begin();
			it_node_ij != LayerNodes->at(layer).end();
			it_node_ij++ )
		{	//��ǰ��㣨�� i �㣩����Ϊ���׽��
			tempFather = (*it_node_ij);
			//=========================================================================
			//	���ݵ�i+1�����Զ� ��i��Ľ���еĴ���� ������������
			//	R�еĴ����  �����½�ֵ����������   ���Ժ���Ľ���½�  >  ǰ�����½�
			//	cmpRP����  ���ݵ� layer �㣨���ԣ���������
			//=========================================================================
			vector<RP*> *node_R = (*it_node_ij)->R;
			// ���Ͻ绹���½����� �� �ڼ������� �� 
			// �½�left ��layer������
			vector<RP*>::iterator low = node_R->begin();
			vector<RP*>::iterator high = node_R-> end();
			high = high - 1;
			QSort(node_R, low, high, layer);
			printNodeR(node_R,layer);
			//std::sort(R->begin(),R->end(), cmpRP);
			
			//=========================================================================
			//	������ i+1 ���½�㣨�������Ĳ��� �� ���׽�㣩
			//	���ϲ㸸�׽ڵ� ���������Ĵ���㼯�ϵ� ��һ�� �����    ���뵽�ýڵ���
			//=========================================================================
			Node *node_ij_1 = new Node(layer + 1, tempFather);
			tempFather->add_SonNode( node_ij_1 );//���׽����Ӻ��ӽ��
			node_ij_1->add_RP( *(node_R->begin()) );
			//=========================================================================
			//	ͨ�����������ƶȶԵ� i+1 ������л���
			//=========================================================================
			vector<RP*>::iterator it_rp;
			bool similarity = true;
			Node *now_Node = node_ij_1;//��ǰ�����Ľ��
			for (it_rp = node_R->begin() + 1; it_rp != node_R->end(); it_rp++)
			{
				//=========================================================================
				// ����������   ʹ��Definition 5
				//=========================================================================
				//similarity = isSimilarity(now_Node, *it_rp,layer);
				now_Node->computeLowAndHigh(layer);
				similarity = isSimilarityNR(now_Node, *it_rp, layer);
				//�ж�������
				if (similarity)
				{	//�����ͽ������ �򽫴������뵽�������ļ���֮��
					now_Node->add_RP(*it_rp);
				}
				else//�����ͽ�㲻���� �򴴽��µĽ��
				{
					//=========================================================================
					// ����ý�㣨now_Node���� �ĵ�layer���Ե�low �� highֵ
					// �����뵽���ĵ� layer ��
					//=========================================================================
					now_Node->computeLowAndHigh( layer );
					// ���뵽�������ĵ� layer+1 ��
					((*LayerNodes)[layer + 1]).insert(now_Node);
					//=========================================================================
					// ����һ���µĽ�� �������Ĳ��� �� ���׽��
					// ������������ĳ�ʼ�����
					//=========================================================================
					Node *node_ij_2 = new Node(layer + 1, tempFather);
					//���ϲ㸸�׽ڵ� ���������Ĵ���㼯�ϵĵ�һ�������    ���뵽�ýڵ���
					node_ij_2->add_RP(*it_rp);
					tempFather->add_SonNode(node_ij_2);//���׽��tempFather��� node_ij_2Ϊ���ӽ��
					now_Node = node_ij_2;//����ǰ��������޸�Ϊ node_ij_2
				}
			}
			if (similarity)
			{
				//=========================================================================
					// ����ý�㣨now_Node���� �ĵ�layer���Ե�low �� highֵ
					// �����뵽���ĵ� layer ��
					//=========================================================================
				now_Node->computeLowAndHigh(layer);
				// ���뵽�������ĵ�layer��
				((*LayerNodes)[layer + 1]).insert(now_Node);
			}
			else 
			{
				vector<Node*>::iterator finalson = tempFather->sons_Node->end() - 1;
				(*finalson)->computeLowAndHigh(layer);
				// ���뵽�������ĵ�layer��
				((*LayerNodes)[ layer + 1 ]).insert(*finalson);
				//node_ij_2->computeLowAndHigh(layer);
			}
		}
		layer++;
		// ����ѭ������
		lastLayer = getINodesI( &(*LayerNodes)[layer-1] );
		curLayer = getINodesI( &(*LayerNodes)[layer] );
	}

}

/*
����r_wait���ھӽ��
input:
	Root: �������ĸ����
	r_wait: �¼���Ĵ����
	threshold: ��ֵ�������жϾ���
Output:
	R_neighbor: r_wait�������ھӼ���
	vector<Node*> *Path: ����·��
*/
void FindingNeighbors(Node* Root, RP* r_wait, double threshold, int attrs, 
	vector<RP*>* retR_nei, vector<Node*> *Path)
{
	printTree(Root);
	//=========================================================================
	// layer ��ʾ��ǰ���� ��ѯ���ĵ�layer��    ������ڵ�0��
	// similarNode ��ʾ���ƽ�� similarNode[0] ��ʾ��0������ƽ��
	// P ָ������Ľ��
	//=========================================================================
	//���� ���ƽ�㹹�ɵĲ㣨���Ʋ㣩 ������
	vector<vector<Node*>*> *similarNode = new vector<vector<Node*>*>();
	Node* P = Root;//ָ��ǰ�����Ľ��
	Root->add_RP(r_wait);
	Path->push_back(P);

	//=========================================================================
	// �������뵽 SimilarNode[0]
	//=========================================================================
	vector<Node*>* rNode = new vector<Node*>();//���ɸ�����
	rNode->push_back(Root);	//�������ѹ��ò�
	similarNode->push_back(rNode);//�� ���Ʋ�0 ���� ����

	//=========================================================================
	// ���������к��ӷ��뵽 SimilarNode[1]
	//=========================================================================
	vector<Node*>* sNode = new vector<Node*>();//��Ÿ����ĺ��ӵ� ���Ʋ�1
	vector<Node*>* son = Root->sons_Node;
	vector<Node*>::iterator it;
	for (it = son->begin(); it != son->end(); it++)
	{
		sNode->push_back(*it);
	}
	similarNode->push_back(sNode);//�� ���Ʋ�1 ��������
	//=========================================================================
	// �ж����ĵ�layer������ƽ�㼯�ϵ�size�Ƿ�Ϊ0
	// layer һ��һ��Ĵ���
	//=========================================================================
	int similarLayer = 1;	//�� 0 ���Ǹ�������ڵĲ㣬�ӵ� 1 �㿪ʼ(���Ʋ�)
	int attrlayer = 0;
	while ((similarNode[similarLayer]).size() != 0)
	{
		//ȡ����ǰ������ƽ�㼯������Ҫ�����Ľ�㣨��similarNode�У�
		vector<Node*>* curLayerS = (*similarNode)[similarLayer];
		Node* newNode = new Node(similarLayer, NULL);// ����һ���½��
		newNode->add_RP(r_wait);//��r_wait���뵽�½ڵ���
		newNode->computeLowAndHigh(attrlayer);//�������Low��Highֵ
		bool existSimilarity = false;// һ���ж��Ƿ�������Ƶı�־����newNode �� ���Ƽ����н�� �Ƚϣ�
		//=========================================================================
		// ( 1 )
		//=========================================================================
		//�洢��newNode���ƽ��ĺ��ӽ��
		vector<Node*>* ChildNode = new vector<Node*>();
		for (it = curLayerS->begin(); it != curLayerS->end(); it++)
		{
			//=========================================================================
			// ����������   ʹ��Definition 5
			//=========================================================================
			Node *curNode = *it;
			//�����������������ԣ���������������������������������
			//bool similarity = isSimilarityNN(curNode, newNode, attrlayer);
			bool similarity = isSimilarityNN(curNode,newNode);
			//�ж�������
			if (similarity)
				//=========================================================================
				// �����һ���������
				// �����ͽ������ �򽫴������뵽�������ļ���֮�У�newNode --�ںϵ�--> curNode��
				// ��ʱ���ǲ���Ӧ�� ��curNode����ΪnewNode��   ���ǵ�
				//=========================================================================
			{
				merge(curNode, newNode);//��newNode�ںϵ�curNode�� �ںϺ� 
										// ��Ҫ���¼��������½�(merge�ڲ������)
				newNode = curNode;		//�ںϺ��curNode��Ϊ�½ڵ�
				vector<Node*>* sons = curNode->sons_Node;
				vector<Node*>::iterator it_s;
				//�� curNode �����к��ӽ��ŵ�ChildNode��
				for (it_s = sons->begin(); it_s != sons->end(); it_s++)
				{
					ChildNode->push_back(*it_s);
				}
				existSimilarity = true;//����������ƽ��
			}
		}
		//=========================================================================
		// �����������ʱ�� 
		// �ӵ�ǰ��㿪ʼ����һ������������
		// �����뵽�� if ����� ��ôִ����Ϻ����ֱ������������
		//=========================================================================
		if (existSimilarity == false)
		{//����һ��ȫ������֧·���ӵ�ǰ�ڵ㿪ʼ��
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
		// ·�� �� ���� 
		//=========================================================================
		P = newNode;		//����ǰ���Pָ�� newNode
		Path->push_back(P);	//���ҷ���·��Path�� ��Path���㷨�ĵ�ʱ���ʹ�ã�
		//printNode(ChildNode);
		std::sort(ChildNode->begin(), ChildNode->end(), cmpNN);// ������ ʹ������
		//printNode(ChildNode);
		vector<Node*>* simiNodes = new vector<Node*>();// �µ� similarNode �е�һ������
		//=========================================================================
		// ( 2 ) ������һ��ѭ���� ���ƽ���
		//=========================================================================
		//ȡ�����ChildNode�е�һ�����

		if (ChildNode->size() != 0)
		{
			Node* new_Node = ChildNode->at(0);
			Node *pNode;
			vector<Node*>::iterator it_s;//������
			// (1) (2) (3) (4) (5) (6) (7) ; 
			// pNode=(1)  nextNode = (2)
			// pNode , nextNode ���ں�(����)����pNode = (1,2)��nextNode=(3)
			// pNode , nextNode �����ں�(������)����pNode = (2)��nextNode=(3)
			for (it_s = ChildNode->begin() + 1; it_s != ChildNode->end(); it_s++)
			{
				pNode = *it_s;
				//bool similarity = isSimilarityNN(pNode, nextNode, attrlayer);
				bool similarity = isSimilarityNN(new_Node, pNode);
				//�ж�������
				if (similarity)
				{	//��pNode�ںϵ�newNode��
					// merge���� �ںϵ������еĽ��   ����Ӱ�쵽ChildNode
					merge(new_Node,pNode);//��������������������������������������������������������
				}
				else
				{//��֤ pNode �� nextNode ��������
					simiNodes->push_back(new_Node);//�� ChildNode�н�� ���뵽 �µ����Ʋ�
					new_Node = pNode;//
				}
			}
			//�� ChildNode�н�� ���뵽 �µ����Ʋ�
			simiNodes->push_back(new_Node);
			//similarNode���һ�����ƽ���
			similarNode->push_back(simiNodes);
		}
		//=========================================================================
		// ( 4 ) ���˲�ѯ�������һ���ʱ��
		// ������ newNode����еĴ�������ĺ�r_wait�ľ��� 
		// ���䱣����r_wait���ھӴ���㼯����
		//=========================================================================
		if (similarLayer == attrs)
		{
			vector<RP*> *rps = newNode->R;	//��ȡ newNode �����Ĵ���㼯��
			vector<RP*>::iterator it_r;		//������
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
���¾���
	�㷨3���㷨4�б�ѭ������
input:
	R: ����㼯�ϣ������㷨1��
	G: ����������ͼ�������㷨1��
	U2: �¼����¼����
	clusters: ���༯��
	attrs:������Ŀ
	alpha��
	beta:
	threshold: ��ֵ�������жϾ���
Output:
	clusters: ���༯��
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
		RP* r_wait = *R_itor;							//ȡ�������ָ��
		vector<RP*>* R_neighbor = new vector<RP*>();	//r_wait �������ھӴ���㼯��
		vector<Node*> *Path = new vector<Node*>();
		//-------------------------------------------------------------------????????????
		//��1��=========================================================================
		//	��ȡr_wait���ھӽ�㼯�ϣ������㷨����
		//=========================================================================
		
		FindingNeighbors(root, r_wait, threshold, attrs, R_neighbor, Path);
		cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
		vector<int>::iterator it_x;		//������ ���ڵ���r_wait�ĸ�����Ķ���
		vector<int>* cover = (r_wait->Cover);
		//�洢û��ӳ�䵽R_neighbor�ļ�¼|����
		vector<double> *noMapping = new vector<double>();
		//��2��=========================================================================
		// �����ж� ��r_wait�еļ�¼Or���� �Ƿ�ӳ�䵽 R_neighbor�е��ھӴ����
		// �������ӳ��		��r_wait�а����Ķ����ŵ���Ӧӳ������ 
		// ������ӳ��		
		//=========================================================================
		bool flag;
		//��3��=========================================================================
		// �ж�r_wait������м�¼x �� r_wait���ھӴ����ľ���dist
		// dist  <  ��ֵ�� �򽫼�¼x ���뵽 ��Ӧ��r_wait���ھӴ���� �ĸ�����Cover��
		//			���� ����¼x���뵽�����ӳ�����ļ���noMapping��
		//=========================================================================
		vector<RP*>::iterator it_rp;
		for (it_x = cover->begin(); it_x != cover->end(); 
			it_x++)
		{//r_wait�ĸ�����Ķ���: it_x
			double x = *it_x;
			flag = true;//
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{//r_wait ���ھӴ����: rp
				RP* rp = *it_rp;
				double dist = getDistanceRX(x, rp, U, attrs);//��ȡ����
				if (dist <= threshold)//�ж�����
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
		//��4��=========================================================================
		// ������������һ�������r_wait����� ������������û���غ�ʱ��
		// �ж� r_wait �����ھӴ�����ж���x �ľ��� �Ƿ�С����ֵ
		// С�� ��ֵ ʱ�򣬽�x�ŵ�r_wait��������ȥ
		//=========================================================================
		if (noMapping->size() == 0)
		{
			//���� r_wait ���ھӴ����
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				vector<int>* cover2 = (rp->Cover);
				// �����ھӴ�����еļ�¼(����)
				for (it_x = cover2->begin(); it_x != cover2->end(); it_x++)
				{
					double x = *it_x;
					//���� r_wait  �� �ھӵ����֮��ľ���
					double dist = getDistanceRX(x, r_wait, U, attrs);
					if (dist <= threshold)
					{
						//Cover(r_wait) = Cover(r_wait) U x
						//�ھӵ������뵽 r_wait �����ĸ�����
						r_wait->CoverPush(x);
					}
				}
			}
			//��r_wait����� ���뵽 R_neighbor
			R_neighbor->push_back(r_wait);
		}
		//��5��=========================================================================
		//	���������ǰ���������r_wait����� �͸������������Ĳ���Or��ȫ�غ�ʱ��
		//	�ӣ��㷨3������Path·����ɾ�� ����е� r_wait �����
		//=========================================================================
		else
		{
			vector<RP*>::iterator it_rp2;
			//����Path·���еĲ�ѯ�����
			for (it_node = Path->begin(); it_node != Path->end(); it_node++)
			{
				Node* node = *it_node;
				//ȡ����ѯ�����Ĵ���㼯��
				vector<RP*> *R = node->R;
				for (it_rp2 = R->begin(); it_rp2 != R->end(); it_rp2++)
				{
					//ɾ��·���в�ѯ�����֮�е� r_wait �����
					if (*it_rp2 == r_wait)
					{
						R->erase(it_rp2);
						break;
					}
				}
			}
		}

		//��6��=========================================================================
		// ������ͼ ��������������������������������
		// ����r_wait���ھӴ���� ������Χ����������
		// ---------------------------------------------
		// ����   ������  
		// r_wait����� ���뵽 R_neighbor�� ===�����������µĴ�������� ��4��
		// ��ôҪ�� G ������r_wait �������к���
		//=========================================================================
		double similarityRiRj;
		//���� r_wait ���ھӴ����  ����  
		vector<RP*> *neighbor;
		vector<RP*>::iterator it_rp_n;
		for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
		{
			//��ȡ r_wait���ھӴ���� ���ھӼ���neighbor
			neighbor = ((*it_rp)->rpNeighbor);
			for (it_rp_n = neighbor->begin(); it_rp_n != neighbor->end(); it_rp_n++)
			{
				RP* ri = *it_rp;
				RP* rj = *it_rp_n;
				//��ȡ�������������ƶ�
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
	//��7��=========================================================================
	// ��ȡ���յľ�����
	//=========================================================================
	int newN = 1;
	list<vector<int>*>::iterator it; //����һ�������� ���ڵ���Strong link��
	list<set<int>*>::iterator itW; //����һ�������� ���ڵ���weak link��
	for (it = G->subGraphs.begin(), itW = G->subGraphsWeak.begin();
		it != G->subGraphs.end();
		it++, itW++) {
		Cluster *Cx = new Cluster(newN);

		for (int i = 0; i < (*it)->size(); i++)
		{
			//���� �洢�ľ����л����е�λ��      ӳ�䵽    ������ָ��
			RP* rp = G->getMapping((*(*it))[i]);//(*(*it))[i]    (*it)->at(i)
			Cx->addCoverToPOS(rp);
		}
		set<int>::iterator itW2;
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//���� �洢�ľ����л����е�λ��      ӳ�䵽    ������ָ��
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