import { ImageOutlined } from '@material-ui/icons'
import React from 'react'
import {Link} from "react-router-dom";

function Navbar1() {
    return (
        <>
         <div className="navbar">
         <div className="navbar-container">
          <Link to="/" className="navbar-logo">TRAVEL <i className="fab-fa-type3"/></Link>
         </div>
         </div>
        </>
    )
}

export default Navbar1
