import React from "react";
import { NavLink } from "react-router-dom";
import Web from "../src/Images/svgimg3.jpg";

const Common = (props) => {
 return(
  <>
  <section id="header" className="d-flex align-items-center">
  <div className="container-fluid">
  <div className="row">
  <div className="col-10 mx-auto">
  <div className="row">
  <div className="col-md-6 pt-5 pt-lg-0 order-2 order-lg-1 d-flex justify-content-center flex-column">
  <h1>{props.name}
   <strong className="brand-name"> Study*Web </strong>
  <h2> Connect with us for Best Web Learning</h2>
  <div className="mt-3">
    <NavLink to={props.visit} 
    className="btn-get-started">{props.btname}</NavLink>
  </div>
   </h1>
    </div>
    <div className="col-lg-6 order-1 order-lg-2 header-img">
      <img src={props.imgscr} 
      className="img-fluid animated" 
      alt= "home img"/>
    </div>
    </div>
     </div>    
      </div>
        </div>
          </section>
  </>
 );
};
export default Common;