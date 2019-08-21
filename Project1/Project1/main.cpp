
#include<iostream>
#include<fstream>
#include"initStructure.h"
using namespace std;

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
/*指针数组 *arr[]    arr是个常量   数组中存的是指针
*/
void getData(string loc,double *arr[], int &i,int begin=0)
{
	std::ifstream in(loc, ios::in);

	double floa;
	char dou;
	int j = 0;
	i = begin;
	while (in.eof() != true)
	{
		double *ar = (double*)malloc(6 * sizeof(double));
		j = 0;
		in >> floa >> dou;
		ar[j++] = i;
		ar[j++] = floa;

		in >> floa >> dou;
		ar[j++] = floa;

		in >> floa >> dou;
		ar[j++] = floa;

		in >> floa >> dou;
		ar[j++] = floa;

		in >> floa;
		ar[j++] = floa;

		//for (int t = 0; t < 5; t++)
		//	cout << ar[t]<< '\t';
		//cout<< endl;
		arr[i++] = ar;
	}
	in.close();
}

void getData(string loc,vector<vector<double>> *U,int begin=0)
{
	std::ifstream in(loc, ios::in);
	char dou;
	int i,j = 0;
	i = begin;
	while (in.eof() != true)
	{
		vector<double> ar;
		ar.resize(6);
		ar[0] = i++;
		in >> ar[1] >> dou>> ar[2] >> dou >> ar[3] >> dou>> ar[4] >> dou >> ar[5] ;
		U->push_back(ar);
	}
	in.close();
}



void TIOC_TWD_vector()
{
	vector<vector<double>> *U = new vector<vector<double>>();
	int count = 0;
	cout << "==============================================================" << endl;
	cout << "数据预处理中" << endl;
	cout << "==============================================================" << endl;
	getData("DataSet\\Banknote\\test.txt", U);
	//=====================================================================
	// 0 . 处理初始数据（数据预处理）
	//=====================================================================
	printU(U);
	cout << "==============================================================" << endl;
	cout << "数据预处理完成" << endl;
	cout << "==============================================================" << endl;

	vector<RP*> *R = new vector<RP*>();
	Graph *graph;
	vector<Cluster*> *clusters;
	//根结点  在第0层    无父结点
	Node *Root = new Node(0, NULL);

	cout << "==============================================================" << endl;
	cout << "1 . 调用第一个算法 建立矩阵、聚类" << endl;
	cout << "==============================================================" << endl;
	SOC_TWD(U, U->size(), 5, 0.06, 0.03, 2, R, &graph, &clusters);

	cout << "==============================================================" << endl;
	cout << "2 . 生成初始查找树（返回树的根结点）" << endl;
	cout << "==============================================================" << endl;
	CST(R, 5, Root, 0.9);
	//打印显示
	printTree(Root);
	cout << "==============================================================" << endl;
	cout << "3 . 处理更新数据（更新数据预处理）" << endl;
	cout << "==============================================================" << endl;
	vector<vector<double>> *U2 = new vector<vector<double>>();
	int count2 = 0;

	cout << "==============================================================" << endl;
	cout << " 3 . 1 . 更新数据预处理中" << endl;
	cout << "==============================================================" << endl;
	getData("DataSet\\Banknote\\test2.txt", U2, count + 1);
	vector<RP*> *R_update = new vector<RP*>();
	Graph *graph2;
	vector<Cluster*> *clusters2;

	cout << "==============================================================" << endl;
	cout << "4 . 使用SOC_TWD算法获取增加数据的代表点" << endl;
	cout << "==============================================================" << endl;
	SOC_TWD(U2, U2->size(), 5, 0.06, 0.03, 2, R_update, &graph2, &clusters2);
	cout << "==============================================================" << endl;
	cout << "5 . 使用聚类更新算法 更新聚类 （算法四内部调用算法三）" << endl;
	cout << "==============================================================" << endl;
	UpdatingClustering(R_update, 0.06, 0.03, 2,
		count, Root, U2, graph, clusters);

	getchar();
}
void TIOC_TWD_array()
{
	//----------------------------------------------------------------------------------------
	double *arr[2000];
	int count = 0;
	cout << "数据预处理中。。。" << endl;
	getData("DataSet\\Banknote\\test.txt", arr, count);
	//=====================================================================
	// 0 . 处理初始数据（数据预处理）
	//=====================================================================
	for (int i = 0; i < count; i++)
	{
		for (int j = 0; j < 6; j++)
			cout << *(*(arr + i) + j) << '\t';
		cout << endl;
	}
	cout << "数据预处理完成。。。" << endl;
	vector<RP*> *R = new vector<RP*>();
	Graph *graph;
	vector<Cluster*> *clusters;
	//根结点  在第0层    无父结点
	Node *Root = new Node(0, NULL);
	//=====================================================================
	// 1 . 生成初始 矩阵和聚类
	//=====================================================================
	SOC_TWD(arr, count, 5, 0.06, 0.03, 2, R, &graph, &clusters);
	//=====================================================================
	// 2 . 生成初始查找树（返回树的根结点）
	//=====================================================================
	CST(R, 5, Root, 0.9);
	//打印显示
	printTree(Root);
	//=====================================================================
	// 3 . 处理更新数据（更新数据预处理）
	//=====================================================================
	double *arr2[2000];
	int count2 = 0;
	cout << "更新数据预处理中。。。" << endl;
	getData("DataSet\\Banknote\\test2.txt", arr2, count2, count + 1);
	vector<RP*> *R_update = new vector<RP*>();
	Graph *graph2;
	vector<Cluster*> *clusters2;

	//=====================================================================
	// 4 . 使用SOC_TWD算法获取增加数据的代表点
	//=====================================================================
	SOC_TWD(arr2, count2, 5, 0.06, 0.03, 2, R_update, &graph2, &clusters2);
	//=====================================================================
	// 5 . 使用聚类更新算法 更新聚类 （算法四内部调用算法三）
	//=====================================================================
	UpdatingClustering(R_update, 0.06, 0.03, 2,
		count, Root, arr2, graph, clusters);

	getchar();
}

#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
bool cmp(int v, int y)
{
	return v > y;
}
int main()
{
	int arr[] = {516,8,68,56,51};
	std::sort(arr,arr+5,cmp);
	for (int t = 0; t < 5; t++)
		cout << arr[t] <<"   ";

	std::cout << RED << "Red Color" << std::endl;
	std::cout << GREEN << "Green Color"  << std::endl;
	TIOC_TWD_vector();
}