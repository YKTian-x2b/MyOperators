#include <stdio.h>

void iniNrun()
{ 
    //override   
    static int flag = 0;
    if(!flag){
        flag = 1;
        printf("init\n");
    }
    else{
        printf("run\n");
    }     
}

void main() {  
    while(1) {
        iniNrun();
    }  
} 