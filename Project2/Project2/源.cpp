/*
功能：实现猜数字游戏
作者：董常青
版本：V1.0
版权：董常青所有
历史记录：
2019/11/19日完成撰写
*/
#include <iostream>
#include <vector>
#include <string>
#include <Windows.h>

using namespace std;

string Guessnum(vector<int> v1,vector<int> v2)
{
	string res="";
	vector<int> v3,v4;
	int na=0,nb=0;
	int len1 = v1.size();
	for(int i=0;i!=len1;++i)
	{
		if(v1[i] == v2[i])
		{
			na++;
		}
		else
		{
			v3.push_back(v1[i]);
			v4.push_back(v2[i]);
		}
	}
	int len2 = v3.size();
	for(int i=0;i!=len2;++i)
	{
		int t1 = v3[i];
		for(int j=0;j!=len2;++j)
		{
			int t2 = v4[j];
			if(t1==t2)
			{
				nb++;
			}
		}
	}
	char cha = na+48;
	char chb = nb+48;
	res = res  + cha + "A" + chb + "B";
	return res;
}

int main()
{
	vector<int> vec1,vec2;
	for(int i=0;i!=4;++i)
	{
		int t1;
		cin>>t1;
		vec1.push_back(t1);
	}
	for(int j=0;j!=4;++j)
	{
		int t1;
		cin>>t1;
		vec2.push_back(t1);
	}
	string out;
	out = Guessnum(vec1,vec2);
	cout<<out<<endl;
	system("pause");
}