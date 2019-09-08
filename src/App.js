
import React, { Component } from 'react';
import { ReactiveBase, DatePicker } from '@appbaseio/reactivesearch';
import { ReactiveOpenStreetMap } from '@appbaseio/reactivemaps';

import './App.css';

class App extends Component {
    render() {
        return (
            <ReactiveBase
                app="water-flow"
                url="http://localhost:9200"
                type="entry"
            >
                <div
                    style={{
                        width: '100%',
                        display: 'flex',
                        flexDirection: 'row',
                        justifyContent: 'space-between'
                    }}
                >
                    <DatePicker
                        title="Date"
                        componentId="date"
                        dataField="date"
                        size={50}
                        showSearch={true}
                    />

                    <ReactiveOpenStreetMap
                        componentId="map"
                        dataField="location"
                        react={{
                            and: "date"
                        }}
                        size={100}
                        renderData={result => ({
                            label: result.mag
                        })}
                    />
                </div>
            </ReactiveBase>
        );
    }
}

export default App;
