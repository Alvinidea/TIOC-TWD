#include"initStructure.h"
using namespace std;


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
void FindingNeighbors(Node* Root,RP* r_wait,double threshold,int attrs, vector<RP*>* ret, vector<Node*> *Path)
{
	//=========================================================================
	// layer 表示当前操作 查询树的第layer层    根结点在第0层
	// similarNode 表示相似结点 similarNode[0] 表示第0层的相似结点
	// P 指向操作的结点
	//=========================================================================
	int layer = 1;
	vector<vector<Node*>*> *similarNode = new vector<vector<Node*>*>();
	Node* P = Root;//指向当前操作的结点
	Root->add_RP(r_wait);

	//=========================================================================
	// 根结点放入到 SimilarNode[0]
	//=========================================================================
	vector<Node*>* rNode = new vector<Node*>();
	rNode->push_back(Root);
	similarNode->push_back(rNode);

	//=========================================================================
	// 根结点的所有孩子放入到 SimilarNode[1]
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
	// 判断树的第layer层的相似结点集合的size是否为0
	// layer 一层一层的处理
	//=========================================================================
	while ((similarNode[layer]).size() != 0)
	{
		//取出当前层的相似结点集合中需要操作的结点（在similarNode中）
		vector<Node*>* curLayerS = (*similarNode)[layer];
		// 产生一个新结点，并将r_wait存入该节点
		Node* newNode = new Node(layer,NULL);
		newNode->add_RP(r_wait);
		// 一个标志判断是否存在相似（用newNode 和 相似集合中结点 比较）
		bool notExistSimilarity = true;
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
			//计算两个结点的相似性
			bool similarity = isSimilarityNN(curNode, newNode, layer);
			//判断相似性
			if (similarity)
				//=========================================================================
				// 处理第一、二种情况
				// 代表点和结点相似 则将代表点加入到结点包含的集合之中（newNode --融合到--> curNode）
				// 这时候是不是应该 将curNode设置为newNode？？？？？？？？？？？
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
		// 第三种情况的时候 
		// 从当前结点开始构建一条独立的子树
		//=========================================================================
		if (notExistSimilarity == true)
		{//构建一条全新树的支路（从根节点开始）
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
		//将当前结点P指向 newNode ，并且放入路径Path中 （Path在算法四的时候会使用）
		P = newNode;
		Path->push_back(P);

		//----------???????????????????????
		sort(ChildNode->begin(), ChildNode->end(), cmpNN);

		//新的 similarNode 中的一个对象
		vector<Node*>* vNode = new vector<Node*>();
		
		//取出存放ChildNode中第一个
		Node* pNode = ChildNode->at(0);
		Node *nextNode;
		vector<Node*>::iterator it_s;

		for (it_s = ChildNode->begin()+1; it_s != ChildNode->end(); it_s++)
		{
			nextNode = *it_s;
			bool similarity = isSimilarityNN(pNode, nextNode, layer);
			//判断相似性
			if (similarity)
			{	//代表点和结点相似 则将代表点加入到结点包含的集合之中
				merge(pNode, nextNode);
			}
			else
			{//保证pNode 和nextNode结点的相邻
				vNode->push_back(pNode);
				pNode = nextNode;
			}
		}
		//similarNode添加一层相似结点层
		similarNode->push_back(vNode);

		//到了查询树的最后一层的时候
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

		//（1）=========================================================================
		//	获取r_wait的邻居结点集合（调用算法三）
		//=========================================================================
		FindingNeighbors(root, r_wait, threshold, attrs, R_neighbor,Path);

		vector<int>::iterator it_x;
		vector<int>* cover = &(r_wait->Cover);
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
		for (it_x = cover->begin(); it_x != cover->end(); it_x++)
		{
			double x = *it_x;
			flag = true;
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				double dist = getDistanceRX(x, rp, U2, attrs);
				if (dist <= threshold)//判断条件
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
		//（4）=========================================================================
		// 三种情况的最后一种情况（r_wait代表点 和其他代表点的没有重合时候）
		// 判断 r_wait 和其邻居代表点中对象的距离 是否小于阈值
		// 
		//=========================================================================
		if (noMapping->size() == 0)
		{
			//遍历 r_wait 的邻居代表点
			for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
			{
				RP* rp = *it_rp;
				vector<int>* cover2 = &(rp->Cover);
				// 遍历邻居代表点中的记录(对象)
				for (it_x = cover2->begin(); it_x != cover2->end(); it_x++)
				{
					double x = *it_x;
					//计算 r_wait  与 邻居点对象之间的距离
					double dist = getDistanceRX(x, r_wait, U2, attrs);
					if (dist <= threshold)
					{
						//Cover(r_wait) = Cover(r_wait) U x
						//邻居点对象放入到 r_wait 代表点的覆盖域
						r_wait->Cover.push_back(x);
					}
				}
			}
			//将r_wait代表点 放入到 R_neighbor
			R_neighbor->push_back(r_wait);
		}
		//（5）=========================================================================
		//三种情况的前两种情况（r_wait代表点 和附近其他代表点的部分Or完全重合时候）
		//从（算法3产生）Path路径中删除 结点中的 r_wait 代表点
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
		for (it_rp = R_neighbor->begin(); it_rp != R_neighbor->end(); it_rp++)
		{
			//获取 r_wait的邻居代表点 的邻居集合neighbor
			vector<RP*> *neighbor = &((*it_rp)->rpNeighbor->Nei);
			vector<RP*>::iterator it_rp_n;
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
}
