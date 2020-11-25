import React from "react";
import {NavLink} from "react-router-dom";

const Menu = () => {
 return(
  <>
  <div className="header_Menu">
  <NavLink  exact activeClassName="active_css" to="/Home"> HOME </NavLink>
  <NavLink  exact activeClassName="active_css"to="/About"> ABOUT</NavLink>
  {/* <NavLink  exact activeClassName="active_css"to="/User/"> USER</NavLink> */}
  {/* <NavLink  exact activeClassName="active_css"to="/Search/"> SEARCH</NavLink> */}
  <NavLink  exact activeClassName="active_css"to="/Contact"> CONTACT</NavLink>
  </div>
  </>
 )
} 
export default Menu;