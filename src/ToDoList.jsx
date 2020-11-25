import React, { useState } from 'react';
import DeleteForeverIcon from '@material-ui/icons/DeleteForever';
 
const ToDoList = (props) => {
const[line , setLine] = useState(false);
 
const iconDelete = () => {
    setLine(true);
};
 return (
 <>
 <div className = "todo_style">
 <span onClick = {iconDelete}> <DeleteForeverIcon className = "delete_Icon" /></span>
 <li style = {{textDecoration : line ? "line-through": "none"}}> {props.text}  </li>
 </div>
 </>
 );
};

export default ToDoList;