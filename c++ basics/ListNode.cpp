#include <iostream>
#include <string>
using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
int main(){
    ListNode *a,*anode;
    
    
    a = new ListNode(); //分配記憶體且利用constructer建立
    anode = new ListNode(3);
    a->next = anode;

    if(a->next!=NULL){
        cout << "the value of a is:" << a->val <<endl;
        cout << "the value of anode is:" << anode->val <<endl;
        a->next=anode->next;
        
    }    
    
    
    delete(a);
    delete(anode);
}
