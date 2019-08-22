
#include<iostream>
#include<fstream>
#include"initStructure.h"
using namespace std;
#define ATTRIBUTES 5
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


void TIOC_TWD()
{
	vector<vector<double>> *U = new vector<vector<double>>();
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
	SOC_TWD(U, U->size(), ATTRIBUTES, 0.06, 0.03, 2, R, &graph, &clusters);

	cout << "==============================================================" << endl;
	cout << "2 . ���ɳ�ʼ���������������ĸ���㣩" << endl;
	cout << "==============================================================" << endl;
	CST(R, ATTRIBUTES, Root, 0.9);
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
	getData("DataSet\\Banknote\\test2.txt", U2, U->size());
	vector<RP*> *R_update = new vector<RP*>();
	Graph *graph2;
	vector<Cluster*> *clusters2;

	cout << "==============================================================" << endl;
	cout << "4 . ʹ��SOC_TWD�㷨��ȡ�������ݵĴ����" << endl;
	cout << "==============================================================" << endl;
	SOC_TWD(U2, U2->size(), ATTRIBUTES, 0.06, 0.03, 2, R_update, &graph2, &clusters2);
	cout << "==============================================================" << endl;
	cout << "5 . ʹ�þ�������㷨 ���¾��� ���㷨���ڲ������㷨����" << endl;
	cout << "==============================================================" << endl;
	

	vector<vector<double>>::iterator it_U;
	for (it_U = U2->begin(); it_U != U2->end(); it_U++)
		U->push_back(*it_U);
	UpdatingClustering(R_update, 0.06, 0.03, 2,
		ATTRIBUTES, Root, U, graph, clusters);

	getchar();
}

int main()
{

	std::cout << RED << "Red Color" << std::endl;
	std::cout << GREEN << "Green Color"  << std::endl;
	TIOC_TWD();
}