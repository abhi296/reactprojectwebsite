import React from "react";
import { NavLink } from "react-router-dom";

const Navbar = () =>{
    return(
      <>
      <div className="container-fluid nav_bg">
      <div className="row">
      <div className="col-10 mx-auto">
      <nav className="navbar navbar-expand-lg navbar-light bg-light">
  <div className="container-fluid">
  <img className="Logo" src={process.env.PUBLIC_URL+"StudyLogo.jpg"} alt="Logo"/>
    <NavLink className="navbar-brand" to="/">Study*Web</NavLink>
    <button className="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
      <span className="navbar-toggler-icon"></span>
    </button>
    <div className="collapse navbar-collapse" id="navbarSupportedContent">
      <ul className="navbar-nav ml-auto mb-2 mb-lg-0">
        <li className="nav-item">
          <NavLink activeClassName='menu_active' exact className="nav-link active" 
          aria-current="page" to="/">Home</NavLink>
        </li>
        <li className="nav-item">
          <NavLink activeClassName='menu_active' className="nav-link" to="/Service">Services</NavLink>
        </li>
        <li className="nav-item">
          <NavLink activeClassName='menu_active' className="nav-link" to="/About">About</NavLink>
        </li>
        <li className="nav-item">
          <NavLink activeClassName='menu_active' className="nav-link" to="/Contact">Contact</NavLink>
        </li>
      </ul>
    </div>
  </div>
</nav>
</div>
</div>
</div>
      </>
    );
};

export default Navbar;