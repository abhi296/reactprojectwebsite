import React from 'react';
import ReactDOM from 'react-dom';
import "./index.css";

let curDate = new Date();
  curDate = curDate.getHours();
  let Time = "";
const style ={};

if(curDate>=1 && curDate<12){
  Time = "Good Morning";
  style.color = "Red";
}else if(curDate>=12 && curDate<16){
  Time = "Good Afternoon";
  style.color = "Orange";
}else if(curDate>=4 && curDate<20){
  Time = "Good Evening";
  style.color = "Darkblue";
}else{
  Time = "Good Night";
  style.color = "Black";

}
ReactDOM.render(
 <h1>Hello Sir,<span style={style}>{Time}</span></h1>,
document.getElementById("root")
);
