/*文本查询程序
作者：董常青
版本：V1.0
版权：董常青所有
历史：
2019/11/25，开始撰写；
备注：
*/

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <Windows.h>

using namespace std;

class TextQuery
{
public:
	typedef vector<string>::size_type line_no;
	void read_file(ifstream &is)
	{
		store_file(is);
		build_map();
	}
	set<line_no> run_query(const string&) const;
	string text_line(line_no) const;
	void TextQuery::store_file(ifstream &is)
	{
		string textline;
		while(getline(is,textline))
			lines_of_text.push_back(textline);
	}
private:
	void store_file(istream&);
	void build_map();
	vector<string> lines_of_text;
	map<string,set<line_no> > word_map;
};

void print_results(const set<TextQuery::line_no>& locs,const string& sought,const TextQuery &file)
{
	typedef set<TextQuery::line_no> line_nums;
	line_nums::size_type size = locs.size();
	cout<<endl<<sought<<" occurs"<<size<<"\t"<<(size,"time","s")<<endl;
	line_nums::const_iterator it = locs.begin();
	for(;it!=locs.end;++it)
	{
		cout<<"\t(line"<<(*it)+1<<") "<<file.text_line(*it)<<endl;
	}
}

int main()
{
	ifstream fs;
	TextQuery tq;
	tq.read_file(fs);
	while(true)
	{
		cout<<"enter word to look for,or q to quit:";
		string s;
		cin>>s;
		if(!cin || s=="q")
			break;
		set<TextQuery::line_no> locs = tq.run_query(s);
		print_results(locs,s,tq);
	}
	system("pause");
}

