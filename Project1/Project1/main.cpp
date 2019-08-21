
#include<iostream>
#include<fstream>
#include"initStructure.h"
using namespace std;

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
/*ָ������ *arr[]    arr�Ǹ�����   �����д����ָ��
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
	cout << "����Ԥ������" << endl;
	cout << "==============================================================" << endl;
	getData("DataSet\\Banknote\\test.txt", U);
	//=====================================================================
	// 0 . �����ʼ���ݣ�����Ԥ����
	//=====================================================================
	printU(U);
	cout << "==============================================================" << endl;
	cout << "����Ԥ�������" << endl;
	cout << "==============================================================" << endl;

	vector<RP*> *R = new vector<RP*>();
	Graph *graph;
	vector<Cluster*> *clusters;
	//�����  �ڵ�0��    �޸����
	Node *Root = new Node(0, NULL);

	cout << "==============================================================" << endl;
	cout << "1 . ���õ�һ���㷨 �������󡢾���" << endl;
	cout << "==============================================================" << endl;
	SOC_TWD(U, U->size(), 5, 0.06, 0.03, 2, R, &graph, &clusters);

	cout << "==============================================================" << endl;
	cout << "2 . ���ɳ�ʼ���������������ĸ���㣩" << endl;
	cout << "==============================================================" << endl;
	CST(R, 5, Root, 0.9);
	//��ӡ��ʾ
	printTree(Root);
	cout << "==============================================================" << endl;
	cout << "3 . ����������ݣ���������Ԥ����" << endl;
	cout << "==============================================================" << endl;
	vector<vector<double>> *U2 = new vector<vector<double>>();
	int count2 = 0;

	cout << "==============================================================" << endl;
	cout << " 3 . 1 . ��������Ԥ������" << endl;
	cout << "==============================================================" << endl;
	getData("DataSet\\Banknote\\test2.txt", U2, count + 1);
	vector<RP*> *R_update = new vector<RP*>();
	Graph *graph2;
	vector<Cluster*> *clusters2;

	cout << "==============================================================" << endl;
	cout << "4 . ʹ��SOC_TWD�㷨��ȡ�������ݵĴ����" << endl;
	cout << "==============================================================" << endl;
	SOC_TWD(U2, U2->size(), 5, 0.06, 0.03, 2, R_update, &graph2, &clusters2);
	cout << "==============================================================" << endl;
	cout << "5 . ʹ�þ�������㷨 ���¾��� ���㷨���ڲ������㷨����" << endl;
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
	cout << "����Ԥ�����С�����" << endl;
	getData("DataSet\\Banknote\\test.txt", arr, count);
	//=====================================================================
	// 0 . �����ʼ���ݣ�����Ԥ����
	//=====================================================================
	for (int i = 0; i < count; i++)
	{
		for (int j = 0; j < 6; j++)
			cout << *(*(arr + i) + j) << '\t';
		cout << endl;
	}
	cout << "����Ԥ������ɡ�����" << endl;
	vector<RP*> *R = new vector<RP*>();
	Graph *graph;
	vector<Cluster*> *clusters;
	//�����  �ڵ�0��    �޸����
	Node *Root = new Node(0, NULL);
	//=====================================================================
	// 1 . ���ɳ�ʼ ����;���
	//=====================================================================
	SOC_TWD(arr, count, 5, 0.06, 0.03, 2, R, &graph, &clusters);
	//=====================================================================
	// 2 . ���ɳ�ʼ���������������ĸ���㣩
	//=====================================================================
	CST(R, 5, Root, 0.9);
	//��ӡ��ʾ
	printTree(Root);
	//=====================================================================
	// 3 . ����������ݣ���������Ԥ����
	//=====================================================================
	double *arr2[2000];
	int count2 = 0;
	cout << "��������Ԥ�����С�����" << endl;
	getData("DataSet\\Banknote\\test2.txt", arr2, count2, count + 1);
	vector<RP*> *R_update = new vector<RP*>();
	Graph *graph2;
	vector<Cluster*> *clusters2;

	//=====================================================================
	// 4 . ʹ��SOC_TWD�㷨��ȡ�������ݵĴ����
	//=====================================================================
	SOC_TWD(arr2, count2, 5, 0.06, 0.03, 2, R_update, &graph2, &clusters2);
	//=====================================================================
	// 5 . ʹ�þ�������㷨 ���¾��� ���㷨���ڲ������㷨����
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