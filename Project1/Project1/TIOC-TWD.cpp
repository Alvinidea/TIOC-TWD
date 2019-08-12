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
			flag = 1;
			return (*neighbors)[i]->Nei;
		}

	}
}

/*
-- ���������֮���ŷ�Ｘ�¾���
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
������������������ƶ�
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
		while (low < high && cmpRP_Big(pivokey, *high, layer)) high--; // ���˴�Ϊ *high >= pivokey; �����5 8 5�����п���������ȻΪ 5 8 5�� ���ܽ�
		// ���б�Ϊ 5 5 8. �����high��low��Ϊ��һ��Ԫ�أ���QSort��iter-1�����Խ�����;
		*low = *high;
		while (low < high && cmpRP_Small(pivokey, *low, layer)) low++;
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
		QSort(ivec, low, iter - 1, layer);
		QSort(ivec, iter + 1, high, layer);
	}
}
//-----------------------------------------------------------------


/*
function���ж�һ��������ĵ�layer������  �Ƿ���Node�еĴ���������
	rp��������
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
		rp ��layer�����Ե��½�ֵ  ����  node�����д�����ĵ�layer�����Ե��½�ֵ
		����Ϊ�Ź����ˣ�
		�����ж�  ��A��rp��layer�����Ե��½�ֵ  ��  ��B��node�����д�����ĵ�layer�����Ե��Ͻ�ֵ �ͺ�
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
	�������㼯���ں�
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
	//��������ں�
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
	R:	�����㼯��
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
//  ��ӡ����������������ʾ��
//####################################################################################
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
		cout << '\t' << "�ھ���Ŀ��" << nei->count << " ; ";
		vector<int> *Nei = nei->Nei;
		for (int j = 0; j < Nei->size(); j++)
		{
			cout << (*Nei)[j] << " , ";
		}
		cout << endl;
	}
}

//��ӡ�����㼯��
void printR(const vector<RP*> *R, int attrs)
{
	cout << "==========================================" << endl;
	cout << "��ӡ�����㼯�ϡ�������������" << endl;
	cout << "==========================================" << endl;
	for (int i = 0; i < R->size(); i++)
	{
		RP* rp = (*R)[i];
		cout << "�����㣺" << rp->representationPoint << endl;
		cout << '\t' << "������";
		for (int j = 0; j < rp->Cover->size(); j++)
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
	for (int i = 1; i < distance->NUM + 1; i++)
		cout << distance->state[i] << "  ";
	cout << endl;
}
//��ӡ������de ͼ
void printG(Graph *graph)
{
	cout << "==========================================" << endl;
	cout << "��ӡ����������ɵ� Strong AND Weakͼ:" << endl;
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
	// vector<int> �д�����ھ����л����е�λ��   ����   ӳ�䵽�������ָ��
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

		cout << '\t' << "��" << itr << "����������ͼ��" << '\t';
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//���� �洢�ľ����л����е�λ��      ӳ�䵽    �������ָ��
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


//####################################################################################
//  �Ĵ��㷨
//####################################################################################
/*
�㷨1��
	U ��¼����
	N ��¼��Ŀ
	attrs ������Ŀ
	alpha
	beta
	threshold ��ֵ
*/
void SOC_TWD(double *U[], int N, int attrs, double alpha, double beta, double threshold)
{
	//----------------------------------------------------------------------------
	//--��ʼ��
	//----------------------------------------------------------------------------
	// R ��Ŵ����������
	vector<RP*> *R = new vector<RP*>();
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
	--����Xi���ھ�
	--���еļ�¼���ھӶ�Ҫ����
	----------------------------------------------------------------------------*/
	for (int i = 1; i < N + 1; i++)
	{

		Neighbor *nei = new Neighbor(i);
		for (int j = 1; j < N + 1; j++)
		{
			if (i == j)			//�ھӲ���Ҫ��������
				;
			else
			{//�ж���������ľ����Ƿ� <= threshold
				if (distance->getDisVal(i, j) <= threshold)
				{//С�ڵ�������뵽�ö�����ھӼ���
					nei->addNeighbor(j);
				}
			}
		}
		//�洢 Xi ���ھ���Ŀ
		nei->count = nei->Nei->size();
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
	-- ���� R �����㼯��,�������д���������ȥ
	-- RP ���� ������   ��������   ������ �� �½�
	----------------------------------------------------------------------------*/
	//�ж� ������� ���Ƿ񻹴���Row������whileѭ���л��м�ɾ��������
	while (distance->getRows() != 0)
	{
		//ѡ���һ�У�������� δɾ�����ݵĵ�һ�У�
		int record = distance->getFirstRecord();
		if (record == -1)//˵���Ѿ�ɾ����
			break;
		// ����ھӼ���
		vector<int> *nei = getNeighborByRecord( neighbors, record);
		// ��ȡnei  ���ݼ�¼�� �� ������м�¼�ھӼ��� �л�ȡ��Ӧ�ھӼ���
		
		//���ɴ����� ���ô�����¼  ��  ��������   ��ʼ������ֵ���½�
		RP *rp = new RP(record, &nei, attrs);

		for (int i = 0; i < rp->Cover->size(); i++)
		{
			int reco = (*(rp->Cover))[i];
			//���ô������ ������|�½�
			for (int attr = 0; attr < attrs; attr++)
			{
				//Cover�б�����Ƕ����  +1��һ�д���Ǽ�¼��
				// U �Ǵ� 0 ��ʼ����   ������Ŵ� 1 ��ʼ�� ������*(U + reco -1)
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
	-- ����Gͼ   ����   �������ƶ�
	-- ʹ�ô����� �� ��ͼ�еĽ��
	//--------------------------------------------------------------------------*/
	Graph *G = new Graph(R);
	for (int i = 0; i < R->size(); i++)
	{
		RP* ri = (*R)[i];
		double similarityRiRj;//�洢���ƶ�
		//RNeighbor* rnei = ri->rpNeighbor;
		vector<RP*> *rnei = ri->rpNeighbor;
		for (int j = 0; j < R->size(); j++)
		{
			RP* rj = R->at(j);
			if (i == j)
				;
			else
			{
				//��ô�����֮��ľ���
				double distance = getRPDistance(ri, rj, attrs);
				if (distance <= 5 * threshold)
				{
					rnei->push_back(rj);
				}
				//�����������֮������ƶ�
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
			//���� �洢�ľ����л����е�λ��      ӳ�䵽    �������ָ��
			RP* rp = G->getMapping((*(*it))[i]);//(*(*it))[i]    (*it)->at(i)
			Cx->addCoverToPOS(rp);
		}
		set<int>::iterator itW2;
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//���� �洢�ľ����л����е�λ��      ӳ�䵽    �������ָ��
			RP* rp = G->getMapping(*itW2);
			Cx->addCoverToBND(rp);
		}
		newN++;
		clusters->push_back(Cx);
	}
	printCluster(clusters);
}


/*
�㷨2��
Creating the searching tree
	:ʹ�ò���ķ�ʽ����searching tree
input ��
	R�������㼯��
	A������
	threshold����ֵ
output��
	root��root of researching tree
*/
void CST(vector<RP*> *R, int *A,int attrs, double threshold)
{
	/*
		vector<set<Node*>>					������в�� ��㼯��
		set<Node*>			Node(i)			ĳһ��Ľ�㼯��
		Node				node			һ�����
	*/
	vector<set<Node*>> *LayerNodes = new vector<set<Node*>>();
	int layer = 0;
	// �����ĸ����ΪNULL
	Node *Root = new Node(layer,NULL);
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
		// ������ i ���е����н��(һ����� �� �������������)
		//=========================================================================
		set<Node*>::iterator it_node_ij;
		for ( it_node_ij = LayerNodes->at(layer).begin();
			it_node_ij != LayerNodes->at(layer).end();
			it_node_ij++ )
		{	//��ǰ��㣨�� i �㣩����Ϊ���׽��
			tempFather = (*it_node_ij);
			//=========================================================================
			//	���ݵ�i+1�����Զ� ��i��Ľ���еĴ����� ������������
			//	R�еĴ�����  �����½�ֵ����������   ���Ժ���Ľ���½�  >  ǰ�����½�
			//	cmpRP����  ���ݵ� layer �㣨���ԣ���������
			//=========================================================================
			vector<RP*> *R = (*it_node_ij)->R;
			// ���Ͻ绹���½����� �� �ڼ������� �� 
			// �½�left ��layer������
			vector<RP*>::iterator low = R->begin();
			vector<RP*>::iterator high = R -> end();
			QSort(R, low, high, layer);

			//std::sort(R->begin(),R->end(), cmpRP);
			
			//=========================================================================
			//	������ i+1 ���½�㣨�������Ĳ��� �� ���׽�㣩
			//	���ϲ㸸�׽ڵ� ���������Ĵ����㼯�ϵ� ��һ�� ������    ���뵽�ýڵ���
			//=========================================================================
			Node *node_ij_1 = new Node(layer, tempFather);
			tempFather->add_SonNode( node_ij_1 );
			node_ij_1->add_RP( *(R->begin()) );
			//=========================================================================
			//	ͨ������������ƶȶԵ� i+1 ������л���
			//=========================================================================
			vector<RP*>::iterator it_rp;
			for (it_rp = R->begin() + 1; it_rp != R->end(); it_rp++)
			{
				//=========================================================================
				// ����������   ʹ��Definition 5
				//=========================================================================
				bool similarity = isSimilarity(node_ij_1, *it_rp,layer);
				//�ж�������
				if (similarity)
				{	//������ͽ������ �򽫴�������뵽�������ļ���֮��
					node_ij_1->add_RP(*it_rp);
				}
				else//������ͽ�㲻���� �򴴽��µĽ��
				{
					//=========================================================================
					// ����ý�㣨node_ij_1���� �ĵ�layer���Ե�low �� highֵ
					// �����뵽���ĵ� layer ��
					//=========================================================================
					node_ij_1->computeLowAndHigh(layer);
					// ���뵽�������ĵ�layer��
					((*LayerNodes)[layer]).insert(node_ij_1);
					//=========================================================================
					// ����һ���µĽ�� �������Ĳ��� �� ���׽��
					// ������������ĳ�ʼ������
					//=========================================================================
					Node *node_ij_2 = new Node(layer, tempFather);
					//���ϲ㸸�׽ڵ� ���������Ĵ����㼯�ϵĵ�һ��������    ���뵽�ýڵ���
					node_ij_2->add_RP(*it_rp);
					//���׽��tempFather���� node_ij_2Ϊ���ӽ��
					tempFather->add_SonNode(node_ij_2);
				}
			}
		}
		//=========================================================================
		// ����һ���²�
		//=========================================================================
		set<Node*> Nodes;
		LayerNodes->push_back(Nodes);
		
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
	r_wait: �¼���Ĵ�����
	threshold: ��ֵ�������жϾ���
Output:
	R_neighbor: r_wait��������ھӼ���
	vector<Node*> *Path: ����·��
*/
void FindingNeighbors(Node* Root, RP* r_wait, double threshold, int attrs, vector<RP*>* ret, vector<Node*> *Path)
{
	//=========================================================================
	// layer ��ʾ��ǰ���� ��ѯ���ĵ�layer��    ������ڵ�0��
	// similarNode ��ʾ���ƽ�� similarNode[0] ��ʾ��0������ƽ��
	// P ָ������Ľ��
	//=========================================================================
	int layer = 1;
	vector<vector<Node*>*> *similarNode = new vector<vector<Node*>*>();
	Node* P = Root;//ָ��ǰ�����Ľ��
	Root->add_RP(r_wait);

	//=========================================================================
	// �������뵽 SimilarNode[0]
	//=========================================================================
	vector<Node*>* rNode = new vector<Node*>();
	rNode->push_back(Root);
	similarNode->push_back(rNode);

	//=========================================================================
	// ���������к��ӷ��뵽 SimilarNode[1]
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
	// �ж����ĵ�layer������ƽ�㼯�ϵ�size�Ƿ�Ϊ0
	// layer һ��һ��Ĵ���
	//=========================================================================
	while ((similarNode[layer]).size() != 0)
	{
		//ȡ����ǰ������ƽ�㼯������Ҫ�����Ľ�㣨��similarNode�У�
		vector<Node*>* curLayerS = (*similarNode)[layer];
		// ����һ���½�㣬����r_wait����ýڵ�
		Node* newNode = new Node(layer, NULL);
		newNode->add_RP(r_wait);
		// һ����־�ж��Ƿ�������ƣ���newNode �� ���Ƽ����н�� �Ƚϣ�
		bool notExistSimilarity = true;
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
			//������������������
			bool similarity = isSimilarityNN(curNode, newNode, layer);
			//�ж�������
			if (similarity)
				//=========================================================================
				// ������һ���������
				// ������ͽ������ �򽫴�������뵽�������ļ���֮�У�newNode --�ںϵ�--> curNode��
				// ��ʱ���ǲ���Ӧ�� ��curNode����ΪnewNode����������������������
				//=========================================================================
			{
				merge(curNode, newNode);
				newNode = curNode;//�ںϺ��curNode��Ϊ�½ڵ�
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
		// �����������ʱ�� 
		// �ӵ�ǰ��㿪ʼ����һ������������
		//=========================================================================
		if (notExistSimilarity == true)
		{//����һ��ȫ������֧·���Ӹ��ڵ㿪ʼ��
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
		//����ǰ���Pָ�� newNode �����ҷ���·��Path�� ��Path���㷨�ĵ�ʱ���ʹ�ã�
		P = newNode;
		Path->push_back(P);

		//----------???????????????????????������
		sort(ChildNode->begin(), ChildNode->end(), cmpNN);

		//�µ� similarNode �е�һ������
		vector<Node*>* vNode = new vector<Node*>();
		//=========================================================================
		// ( 2 )
		//=========================================================================
		//ȡ�����ChildNode�е�һ��
		Node* pNode = ChildNode->at(0);
		Node *nextNode;
		vector<Node*>::iterator it_s;

		for (it_s = ChildNode->begin() + 1; it_s != ChildNode->end(); it_s++)
		{
			nextNode = *it_s;
			bool similarity = isSimilarityNN(pNode, nextNode, layer);
			//�ж�������
			if (similarity)
			{	//������ͽ������ �򽫴�������뵽�������ļ���֮��
				merge(pNode, nextNode);
			}
			else
			{//��֤pNode ��nextNode��������
				vNode->push_back(pNode);
				pNode = nextNode;
			}
		}
		//similarNode����һ�����ƽ���
		similarNode->push_back(vNode);

		//���˲�ѯ�������һ���ʱ��
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
���¾���
	�㷨3���㷨4�б�ѭ������
input:
	R: �����㼯�ϣ������㷨1��
	G: �����������ͼ�������㷨1��
	U2: �¼����¼����
	clusters: ���༯��
	attrs:������Ŀ
	alpha��
	beta:
	threshold: ��ֵ�������жϾ���
Output:
	clusters: ���༯��
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

		//��1��=========================================================================
		//	��ȡr_wait���ھӽ�㼯�ϣ������㷨����
		//=========================================================================
		FindingNeighbors(root, r_wait, threshold, attrs, R_neighbor, Path);

		vector<int>::iterator it_x;
		vector<int>* cover = r_wait->Cover;
		//�洢û��ӳ�䵽R_neighbor�ļ�¼|����
		vector<double> *noMapping = new vector<double>();
		//��2��=========================================================================
		// �����ж� ��r_wait�еļ�¼Or���� �Ƿ�ӳ�䵽 R_neighbor�е��ھӴ�����
		// �������ӳ��		��r_wait�а����Ķ����ŵ���Ӧӳ������� 
		// ������ӳ��		
		//=========================================================================
		bool flag;

		//��3��=========================================================================
		// �ж�r_wait�������м�¼x �� r_wait���ھӴ�����ľ���dist
		// dist  <  ��ֵ�� �򽫼�¼x ���뵽 ��Ӧ��r_wait���ھӴ����� �ĸ�����Cover��
		//			���� ����¼x���뵽�����ӳ�����ļ���noMapping��
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
				if (dist <= threshold)//�ж�����
				{
					rp->Cover->push_back(x);
					flag = false;
				}
			}
			if (flag)
			{
				noMapping->push_back(x);
			}
		}
		//��4��=========================================================================
		// ������������һ�������r_wait������ �������������û���غ�ʱ��
		// �ж� r_wait �����ھӴ������ж���ľ��� �Ƿ�С����ֵ
		// 
		//=========================================================================
		if (noMapping->size() == 0)
		{
			//���� r_wait ���ھӴ�����
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				vector<int>* cover2 = rp->Cover;
				// �����ھӴ������еļ�¼(����)
				for (it_x = cover2->begin(); it_x != cover2->end(); it_x++)
				{
					double x = *it_x;
					//���� r_wait  �� �ھӵ����֮��ľ���
					double dist = getDistanceRX(x, r_wait, U2, attrs);
					if (dist <= threshold)
					{
						//Cover(r_wait) = Cover(r_wait) U x
						//�ھӵ������뵽 r_wait ������ĸ�����
						r_wait->Cover->push_back(x);
					}
				}
			}
			//��r_wait������ ���뵽 R_neighbor
			R_neighbor->push_back(r_wait);
		}
		//��5��=========================================================================
		//���������ǰ���������r_wait������ �͸�������������Ĳ���Or��ȫ�غ�ʱ��
		//�ӣ��㷨3������Path·����ɾ�� ����е� r_wait ������
		//=========================================================================
		else
		{
			vector<RP*>::iterator it_rp2;
			//����Path·���еĲ�ѯ�����
			for (it_node = Path->begin(); it_node != Path->end(); it_node++)
			{
				Node* node = *it_node;
				//ȡ����ѯ�����Ĵ����㼯��
				vector<RP*> *R = node->R;
				for (it_rp2 = R->begin(); it_rp2 != R->end(); it_rp2++)
				{
					//ɾ��·���в�ѯ�����֮�е� r_wait ������
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
		// ����r_wait���ھӴ����� ������Χ�����������
		// ---------------------------------------------
		// ����   ������  
		// r_wait������ ���뵽 R_neighbor�� ===�����������µĴ��������� ��4��
		// ��ôҪ�� G ������r_wait ��������к���
		//=========================================================================
		double similarityRiRj;
		//���� r_wait ���ھӴ�����  ����  
		for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
		{
			//��ȡ r_wait���ھӴ����� ���ھӼ���neighbor
			//vector<RP*> *neighbor = &((*it_rp)->rpNeighbor->Nei);
			vector<RP*> *neighbor = ((*it_rp)->rpNeighbor);
			vector<RP*>::iterator it_rp_n;
			for (it_rp_n = neighbor->begin(); it_rp_n != neighbor->end(); it_rp_n++)
			{
				RP* ri = *it_rp;
				RP* rj = *it_rp_n;
				//��ȡ��������������ƶ�
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
			//���� �洢�ľ����л����е�λ��      ӳ�䵽    �������ָ��
			RP* rp = G->getMapping((*(*it))[i]);//(*(*it))[i]    (*it)->at(i)
			Cx->addCoverToPOS(rp);
		}
		set<int>::iterator itW2;
		for (itW2 = (*itW)->begin(); itW2 != (*itW)->end(); itW2++) {

			//���� �洢�ľ����л����е�λ��      ӳ�䵽    �������ָ��
			RP* rp = G->getMapping(*itW2);
			Cx->addCoverToBND(rp);
		}
		newN++;
		clusters->push_back(Cx);
	}
}

