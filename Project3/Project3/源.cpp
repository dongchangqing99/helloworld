/*
功能：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
作者：董常青
版本：V1.0
版权：董常青所有
备注：
*/

#include <iostream>
#include <vector>
#include <string>
#include <Windows.h>

using namespace std;

string getch(int i)
{
	string out="";
	switch(i)
	{
	case 2:
		out = "abc";
		break;
	case 3:
		out = "def";
		break;
	case 4:
		out = "ghi";
		break;
	case 5:
		out = "jkl";
		break;
	case 6:
		out = "mno";
		break;
	case 7:
		out = "pqrs";
		break;
	case 8:
		out = "tuv";
		break;
	case 9:
		out = "wxyz";
		break;
	default:
		out = "";
		break;
	}
	return out;
}

vector<string> getstr(vector<string> vecs,string str2)
{
	vector<string> res;
	int len1 = vecs.size();
	int len2 = str2.size();
	cout<<"len1="<<len1<<endl<<"len2="<<len2<<endl;
	if(len1==0)
	{
		for(int k=0;k!=len2;++k)
		{
			string strp = "";
			strp = strp + str2[k];
			res.push_back(strp);
		}
	}
	else
	{
		for(int i=0;i!=len1;++i)
		{
			string st = vecs[i];
			for(int j=0;j!=len2;++j)
			{
				string st2 = st + str2[j];
				res.push_back(st2);
			}
		}
	}
	return res;
}

int main()
{
	string digits;
	std::cin>>digits;
	vector<string> out;
	int len = digits.size();
	vector<int> veci;
	for(int i=0;i!=len;++i)
	{
		char ch = digits[i];
		int t = ch-48;
		veci.push_back(t);
	}
	for(int i=0;i!=veci.size();++i)
	{
		string chi = getch(veci[i]);
		cout<<chi<<endl;
		if(chi.size()>0)
		{
			out = getstr(out,chi);
		}
	}
	
	for(int i=0;i!=out.size();++i)
	{
		cout<<out[i]<<",";
	}

	system("pause");
}