#include<iostream>
#include<vector>
#include <fstream>//包含文件操作头文件
using namespace std;



class preOp {

private :
	int count_inst;
	int count_attr;
	string characteristics;



public :

	void split(std::string str, std::string pattern, vector<string> &result)
	{
		std::string::size_type pos;
		str += pattern;//扩展字符串以方便操作
		int size = str.size();

		for(size_t i=0; i < size; i++)
		{ 0.11,13115,
			pos = str.find(pattern, i);
			if (pos < size)
			{
				std::string s = str.substr(i, pos - i);
				result.push_back(s);
				i = pos + pattern.size(); 
			}
		}
	}
	void transfer(vector<string> &result, vector<float> &row) {}

	void getData(const vector<vector<float>> &vec)
	{
		std::ifstream in("C:\\Users\\ALVIN\\Desktop\\DataSet\\Banknote\\data_banknote_authentication.txt", ios::in);
		char str[50];

		for (int i = 20; i > 0; i--)
		{
			in >> str;
			cout << str;
		}
		in.close();
		

	}


};