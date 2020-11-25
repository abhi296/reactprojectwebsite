import React from "react";
import { NavLink } from "react-router-dom";


const Card = (props) =>{
    return(
    <>
     <div className="col-md-4 col-10 md-show">
  <div className="card">
  <img src={props.imgsrc} className="card-img-top" alt="image"/>
  <div className="card-body">
    <h5 className="card-title font-weight-bold">{props.title}</h5>
    <p className="card-text">Perfect & productive</p>
    <NavLink to="" className="btn btn-primary">Get Started</NavLink>
  </div>
</div>
  </div>
    </>
    );
}
export default Card;