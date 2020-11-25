import React from 'react';
import logo from './Images/logo.webp';
import FindInPageIcon from '@material-ui/icons/FindInPage';

const Header = () => {
  return ( 
  <>
  <div className = "header">
  <img src = {logo}  alt = "logo" className = "logo"></img>
  <h1> Google Notes </h1>
  <FindInPageIcon className = "Icon" color="primary"/>
  </div>
  </>
  );
};

export default Header;