#include"initStructure.h"
using namespace std;


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
void FindingNeighbors(Node* Root,RP* r_wait,double threshold,int attrs, vector<RP*>* ret, vector<Node*> *Path)
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
		Node* newNode = new Node(layer,NULL);
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
				// �����һ���������
				// �����ͽ������ �򽫴������뵽�������ļ���֮�У�newNode --�ںϵ�--> curNode��
				// ��ʱ���ǲ���Ӧ�� ��curNode����ΪnewNode����������������������
				//=========================================================================
			{	
				merge(curNode, newNode);
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
			return ;
			//break;
		}
		//����ǰ���Pָ�� newNode �����ҷ���·��Path�� ��Path���㷨�ĵ�ʱ���ʹ�ã�
		P = newNode;
		Path->push_back(P);

		//----------???????????????????????
		sort(ChildNode->begin(), ChildNode->end(), cmpNN);

		//�µ� similarNode �е�һ������
		vector<Node*>* vNode = new vector<Node*>();
		
		//ȡ�����ChildNode�е�һ��
		Node* pNode = ChildNode->at(0);
		Node *nextNode;
		vector<Node*>::iterator it_s;

		for (it_s = ChildNode->begin()+1; it_s != ChildNode->end(); it_s++)
		{
			nextNode = *it_s;
			bool similarity = isSimilarityNN(pNode, nextNode, layer);
			//�ж�������
			if (similarity)
			{	//�����ͽ������ �򽫴������뵽�������ļ���֮��
				merge(pNode, nextNode);
			}
			else
			{//��֤pNode ��nextNode��������
				vNode->push_back(pNode);
				pNode = nextNode;
			}
		}
		//similarNode���һ�����ƽ���
		similarNode->push_back(vNode);

		//���˲�ѯ�������һ���ʱ��
		if (layer == attrs)
		{
			vector<RP*> *rps = newNode->R;
			vector<RP*>::iterator it_r;
			for (it_r = rps->begin(); it_r != rps->end(); it_r++)
			{
				RP* rp = *it_r;
				if (rp != r_wait && getRPDistance(rp, r_wait,attrs) <= 2 * threshold)
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
void UpdatingClustering(vector<RP*> *R, Graph *graph,double alpha, double beta, double threshold, 
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
		FindingNeighbors(root, r_wait, threshold, attrs, R_neighbor,Path);

		vector<int>::iterator it_x;
		vector<int>* cover = &(r_wait->Cover);
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
					rp->Cover.push_back(x);
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
		// �ж� r_wait �����ھӴ�����ж���ľ��� �Ƿ�С����ֵ
		// 
		//=========================================================================
		if (noMapping->size() == 0)
		{
			//���� r_wait ���ھӴ����
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				vector<int>* cover2 = &(rp->Cover);
				// �����ھӴ�����еļ�¼(����)
				for (it_x = cover2->begin(); it_x != cover2->end(); it_x++)
				{
					double x = *it_x;
					//���� r_wait  �� �ھӵ����֮��ľ���
					double dist = getDistanceRX(x, r_wait, U2, attrs);
					if (dist <= threshold)
					{
						//Cover(r_wait) = Cover(r_wait) U x
						//�ھӵ������뵽 r_wait �����ĸ�����
						r_wait->Cover.push_back(x);
					}
				}
			}
			//��r_wait����� ���뵽 R_neighbor
			R_neighbor->push_back(r_wait);
		}
		//��5��=========================================================================
		//���������ǰ���������r_wait����� �͸������������Ĳ���Or��ȫ�غ�ʱ��
		//�ӣ��㷨3������Path·����ɾ�� ����е� r_wait �����
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
		for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
		{
			//��ȡ r_wait���ھӴ���� ���ھӼ���neighbor
			vector<RP*> *neighbor = &((*it_rp)->rpNeighbor->Nei);
			vector<RP*>::iterator it_rp_n;
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
}
