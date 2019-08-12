/*
input��
	universe  {x1,x2,...,xn}
	��
	��
	cta

output:
	C={{C1_,C1-},...,{Cw_,Cw-}}
	R
	neighbor(ri),ri �� R
	G
*/


/*
U ��¼����
N ��¼��Ŀ
attrs ������Ŀ
alpha 
beta
threshold ��ֵ
*/


void SOC_TWD(double *U[],int N ,int attrs,double alpha,double beta,double threshold)
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
				if (distance->getDisVal(i,j) <= threshold)
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
	
	
	
	
	
	//std::sort(neighbors.begin(), neighbors.end(), cmpNeighbor);
	/*----------------------------------------------------------------------------
	-- ���� R ����㼯��,�������д��������ȥ
	-- RP ���� �����   ��������   ������ �� �½�
	-- 
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
		RP *rp = new RP(record, nei,attrs);
		for (int i = 0; i < rp->Cover.size(); i++)
		{
			int reco = (rp->Cover)[i];
			//���ô����� ������|�½�
			for (int attr = 0; attr < attrs; attr++)
			{
				//Cover�б�����Ƕ����  +1��һ�д���Ǽ�¼��
				double attrVal = *(*(U + reco)+1+attr);
				if (rp->getLeft(attr) > attrVal)
					rp->setLeft(attr,attrVal);
				if (rp->getRight(attr) < attrVal)
					rp->setRight(attr,attrVal);;
			}
			distance->deleteRows(reco);
		}
		R->push_back(rp); 
	}
	/*----------------------------------------------------------------------------
	//----------------------����Gͼ   ����   �������ƶ�--------------------------
	-- ʹ�ô���� �� ��ͼ�еĽ��
	//--------------------------------------------------------------------------*/
	Graph *G = new Graph(R);
	for (int i = 0; i < R->size(); i++)
	{
		RP* ri = (*R)[i];
		double similarityRiRj;//�洢���ƶ�
		RNeighbor* rnei = new RNeighbor(ri);
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
	//----------------------------------------------------------------------------
	//----------------------��ȡǿ��ͨ��ͼ----------------------------------------
	//----------------------------------------------------------------------------
	G->BFS();
	//----------------------------------------------------------------------------
	//----------------------���ɾ��༯��------------------------------------------
	//----------------------------------------------------------------------------
	vector<Cluster*> *clusters = new vector<Cluster*>();
	list<vector<int>*>::iterator it; //����һ�������� ���ڵ���Strong link��
	list<set<int>*>::iterator itW; //����һ�������� ���ڵ���weak link��
	int newN = 1;
	for (it = G->subGraphs.begin(), itW=G->subGraphsWeak.begin();
		it != G->subGraphs.end(); 
		it++,itW++) {
		Cluster *Cx = new Cluster(newN);
		for (int i=0;i< (*it)->size();i++)
		{
			//���� �洢�ľ����л����е�λ��      ӳ�䵽    ������ָ��
			RP* rp = G->getMapping( (*(*it))[i] );//(*(*it))[i]    (*it)->at(i)
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
