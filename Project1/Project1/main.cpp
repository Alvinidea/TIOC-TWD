
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
void getData(double *arr[], int &i)
{
	std::ifstream in("C:\\Users\\ALVIN\\Desktop\\DataSet\\Banknote\\test.txt", ios::in);

	double floa;
	char dou;
	int j = 0;
	i = 0;
	while (in.eof() != true)
	{
		double *ar = (double*)malloc(6 * sizeof(double));
		j = 0;
		in >> floa >> dou;
		ar[j++] = i+1;
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
int main()
{
	double *arr[2000];
	int count = 0;
	cout << "����Ԥ�����С�����" << endl;
	getData(arr, count);

	for (int i = 0; i < count; i++)
	{
		for (int j = 0; j < 6; j++)
			cout << *(*(arr + i) + j) << '\t';
		cout << endl;
	}
	cout << "����Ԥ������ɡ�����" << endl;
	SOC_TWD(arr, count, 5, 0.06, 0.03, 2);
	getchar();
	return 0;
}