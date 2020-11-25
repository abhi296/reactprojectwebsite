import React from "react";
import { Navlink } from "react-router-dom";


const Error = () => {
 return(
  <>
  <div>
  <p className ="error">404 ERROR PAGE</p>
  <h1 className ="new_h1">This Page Dosen't Exist🚫</h1>
  <Navlink to="/">Go Back</Navlink>
  </div>
  </>
 )
} ;
export default Error;