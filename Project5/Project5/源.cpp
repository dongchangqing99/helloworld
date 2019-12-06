/*
功能：删除数组中重复元素
作者：董常青
版本：V1.0
历史记录：
2019/11/21，完成撰写
备注：

*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <Windows.h>

using namespace std;

int main()
{
	vector<int> nums;
	nums.push_back(1);
	nums.push_back(1);
	nums.push_back(3);
	int val = 1;
	vector<int> numo;
	for(int i=0;i!=nums.size();++i)
	{
		if(nums[i]==val)
		{
			continue;
		}
		else
		{
			numo.push_back(nums[i]);
		}
	}
	nums = numo;
	cout<<nums.size()<<endl;

	/*int val = 1;
	int cnt = 0;
	sort(nums.begin(),nums.end());
	for(int i=0;i!=nums.size();++i)
	{
		if(nums[i]==val)
		{
			cnt++;
		}
	}
	vector<int>::iterator it=find(nums.begin(),nums.end(),1);
	nums.erase(it,it+cnt);
	for(int i=0;i!=nums.size();++i)
	{
		cout<<nums[i]<<",";
	}
	cout<<endl<<"len="<<nums.size()<<endl;*/

	//int len = nums.size();
	//int i=0;
	//for(int j=1;j!=len;++j)
	//{
	//	if(nums[i] != nums[j])
	//	{
	//		nums[++i] = nums[j];
	//	}
	//}
	//cout<<i+1<<endl;

	//vector<int> numn;
	//if(len>=1)
	//{
	//sort(nums.begin(),nums.end());
	//int i=0;
	//numn.push_back(nums[0]);
	//for(int i=0;i!=nums.size()-1;++i)
	//{
	//	if(nums[i]!=nums[i+1])
	//	{
	//		numn.push_back(nums[i+1]);
	//	}
	//}
	//}
	//for(int i=0;i!=numn.size();++i)
	//{
	//	cout<<numn[i]<<endl;
	//}

	system("pause");
}