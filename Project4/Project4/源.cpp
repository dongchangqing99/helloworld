/*
功能：给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
作者：董常青
版本：V1.0
版权：董常青所有
备注：
https://blog.csdn.net/sinat_38486449/article/details/80194132,
*/

#include <iostream>
#include <Windows.h>
#include <vector>
#include "Node.h"

using namespace std;

struct Node
{
	int data;
	struct Node *next;
};

Node* creatnode(int cnt)
{
	Node *p1,*p2,*head;
	head = p1 = (Node*)malloc(sizeof(Node));
	head->data = 1;
	head->next = NULL;
	for(int i=1;i!=cnt;++i)
	{
		p2 = (Node*)malloc(sizeof(Node));
		p2->data = i+1;
		p1->next = p2;
		p1 = p2;
	}
	p1->next = NULL;
	return head;
}

int main()
{
	Node *head;
	int num = 5;
	int n = 2;
	head = creatnode(num);
	Node *p1;
	p1 = head;
	int i=0;
	cout<<"前:"<<endl;
	while(p1 != NULL)
	{
		cout<<p1->data<<",";
		p1 = p1->next;
		i++;
	}
	Node *p2 = head;
	Node *slow = head;
	Node *fast = slow;
	while(n--)
		fast = fast->next;
	if(fast == NULL)
	{
		head = head->next;
	}
	else
	{
		while(fast->next != NULL)
		{
			slow = slow->next;
			fast = fast->next;
		}
		slow->next = slow->next->next;
	}
	cout<<endl<<"后:"<<endl;
	Node *p3 = head;
	while(p3!=NULL)
	{
		cout<<p3->data<<",";
		p3 = p3->next;
	}
	system("pause");
}