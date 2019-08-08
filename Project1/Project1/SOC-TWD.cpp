#include"initStructure.h"
using namespace std;
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
	// R ��Ŵ���������
	vector<RP*> *R = new vector<RP*>();
	// neighbors ��Ÿ�����¼�������ھӵ�����
	vector<Neighbor*> neighbors;
	// distance ���������¼֮�����ľ���
	Distance *distance = new Distance(N);

	vector<RNeighbor*> rneighbors;

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
			if (i == j)//�ھӲ���Ҫ��������
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
		nei->count = nei->Nei.size();
		// �������е���ھӼ�
		neighbors.push_back(nei);
	}
	/*----------------------------------------------------------------------------
	-- ��neighbors����
	-- ��ÿ��Xi������|��¼�����ھ���Ŀ��������
	----------------------------------------------------------------------------*/

	printNei(&neighbors);
	std::sort(neighbors.begin(), neighbors.end(), cmpNeighbor);
	printNei(&neighbors);
	/*----------------------------------------------------------------------------
	-- ���� R ����㼯��,�������д��������ȥ
	-- RP ���� �����   ��������   ������ �� �½�
	----------------------------------------------------------------------------*/
	//�ж� ������� ���Ƿ񻹴���Row������whileѭ���л��м�ɾ��������
	while (distance->getRows() != 0)
	{
		//ѡ���һ�У�������� δɾ�����ݵĵ�һ�У�
		int record = distance->getFirstRecord();
		if (record == -1)//˵���Ѿ�ɾ����
			break;
		// ����ھӼ���
		vector<int> nei;
		// ��ȡnei  ���ݼ�¼�� �� ������м�¼�ھӼ��� �л�ȡ��Ӧ�ھӼ���
		getNeighborByRecord(neighbors, record, nei);
		//���ɴ���� ���ô����¼  ��  ��������   ��ʼ������ֵ���½�
		RP *rp = new RP(record, nei, attrs);
		for (int i = 0; i < rp->Cover.size(); i++)
		{
			int reco = (rp->Cover)[i];
			//���ô����� ������|�½�
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
	-- ʹ�ô���� �� ��ͼ�еĽ��
	//--------------------------------------------------------------------------*/
	Graph *G = new Graph(R);
	for (int i = 0; i < R->size(); i++)
	{
		RP* ri = (*R)[i];
		double similarityRiRj;//�洢���ƶ�
		RNeighbor* rnei = ri->rpNeighbor;
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
					rnei->addNeighbor(rj);
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
}


/*
�㷨2��
Creating the searching tree
input ��
	R������㼯��
	A������
	threshold����ֵ
output��
	root��root of researching tree
*/
void CST(vector<RP*> *R, int *A,int attrs, double threshold = 0.9)
{
	/*
		vector<set<Node*>>					������в�� ��㼯��
		set<Node*>			Node(i)			ĳһ��Ľ�㼯��
		Node				node			һ�����
	*/
	vector<set<Node*>> *LayerNodes = new vector<set<Node*>>();
	int layer = 0;
	Node *Root = new Node(layer,NULL);
	Root->R = R;
	double lastLayer, curLayer;

	//����һ���µĲ�  ��0��
	set<Node*> Node0;
	Node0.insert(Root);
	LayerNodes->push_back(Node0);

	lastLayer = 0.1;
	curLayer = getINodesI( &Node0 );

	Node *tempFather;
	while (lastLayer/curLayer < threshold || layer < attrs)
	{
		layer++;
		//=========================================================================
		// ������ i ���е����н��(һ����� �� ������������)
		//=========================================================================
		set<Node*>::iterator it_node_ij;
		for ( it_node_ij = LayerNodes->at(layer).begin();
			it_node_ij != LayerNodes->at(layer).end();
			it_node_ij++ )
		{	//���ø��׽��
			tempFather = (*it_node_ij);
			//=========================================================================
			//	���ݵ�i+1�����Զ� ��i��Ľ���еĴ���� ������������
			//	R�еĴ����  �����½�ֵ����������   ���Ժ���Ľ���½�  >  ǰ�����½�
			//	cmpRP����  ���ݵ� layer �㣨���ԣ���������
			//=========================================================================
			vector<RP*> *R = (*it_node_ij)->R;
			std::sort(R->begin(),R->end(), cmpRP);//��������������������������������������е�
			//=========================================================================
			//	������ i+1 ���½�㣨�������Ĳ��� �� ���׽�㣩
			//	���ϲ㸸�׽ڵ� ���������Ĵ���㼯�ϵĵ�һ�������    ���뵽�ýڵ���
			//=========================================================================
			Node *node_ij_1 = new Node(layer, tempFather);
			tempFather->add_SonNode(node_ij_1);
			node_ij_1->add_RP((*R)[0]);
			//=========================================================================
			// ͨ�����������ƶȶԵ� i+1 ������л���
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
				{	//�����ͽ������ �򽫴������뵽�������ļ���֮��
					node_ij_1->add_RP(*it_rp);
				}
				else//�����ͽ�㲻���� �򴴽��µĽ��
				{
					//=========================================================================
					// ����ý�㣨node_ij_1���� �ĵ�layer���Ե�low �� highֵ
					// �����뵽���ĵ� layer ��
					//=========================================================================
					node_ij_1->computeLowAndHigh(layer);
					//���뵽�������ĵ�layer��
					((*LayerNodes)[layer]).insert(node_ij_1);
					//=========================================================================
					// ����һ���µĽ�� �������Ĳ��� �� ���׽��
					// ������������ĳ�ʼ�����
					//=========================================================================
					Node *node_ij_2 = new Node(layer, tempFather);
					//���ϲ㸸�׽ڵ� ���������Ĵ���㼯�ϵĵ�һ�������    ���뵽�ýڵ���
					node_ij_1->add_RP(*it_rp);
					tempFather->add_SonNode(node_ij_2);
				}
			}
		}
		//=========================================================================
		// ����һ���²�
		//=========================================================================
		set<Node*> Nodes;
		LayerNodes->push_back(Nodes);
		//����ѭ������
		lastLayer = getINodesI( &(*LayerNodes)[layer-1] );
		curLayer = getINodesI( &(*LayerNodes)[layer] );
	}

}
