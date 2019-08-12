/*
input：
	universe  {x1,x2,...,xn}
	α
	β
	cta

output:
	C={{C1_,C1-},...,{Cw_,Cw-}}
	R
	neighbor(ri),ri ∈ R
	G
*/


/*
U 记录矩阵
N 记录数目
attrs 属性数目
alpha 
beta
threshold 阈值
*/


void SOC_TWD(double *U[],int N ,int attrs,double alpha,double beta,double threshold)
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

	vector<RNeighbor*> rneighbors;

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
				if (distance->getDisVal(i,j) <= threshold)
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
	
	
	
	
	
	//std::sort(neighbors.begin(), neighbors.end(), cmpNeighbor);
	/*----------------------------------------------------------------------------
	-- 创建 R 代表点集合,并将所有代表点加入进去
	-- RP 包括 代表点   覆盖区域   属性上 和 下界
	-- 
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
		RP *rp = new RP(record, nei,attrs);
		for (int i = 0; i < rp->Cover.size(); i++)
		{
			int reco = (rp->Cover)[i];
			//设置代表点的 属性上|下界
			for (int attr = 0; attr < attrs; attr++)
			{
				//Cover中保存的是对象号  +1第一行存的是记录号
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
	//----------------------创建G图   并且   计算相似度--------------------------
	-- 使用代表点 当 做图中的结点
	//--------------------------------------------------------------------------*/
	Graph *G = new Graph(R);
	for (int i = 0; i < R->size(); i++)
	{
		RP* ri = (*R)[i];
		double similarityRiRj;//存储相似度
		RNeighbor* rnei = new RNeighbor(ri);
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
					rnei->addNeighbor(rj);
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
	//----------------------------------------------------------------------------
	//----------------------获取强连通子图----------------------------------------
	//----------------------------------------------------------------------------
	G->BFS();
	//----------------------------------------------------------------------------
	//----------------------生成聚类集合------------------------------------------
	//----------------------------------------------------------------------------
	vector<Cluster*> *clusters = new vector<Cluster*>();
	list<vector<int>*>::iterator it; //声明一个迭代器 用于迭代Strong link的
	list<set<int>*>::iterator itW; //声明一个迭代器 用于迭代weak link的
	int newN = 1;
	for (it = G->subGraphs.begin(), itW=G->subGraphsWeak.begin();
		it != G->subGraphs.end(); 
		it++,itW++) {
		Cluster *Cx = new Cluster(newN);
		for (int i=0;i< (*it)->size();i++)
		{
			//根据 存储的矩阵行或列中的位置      映射到    代表点的指针
			RP* rp = G->getMapping( (*(*it))[i] );//(*(*it))[i]    (*it)->at(i)
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
